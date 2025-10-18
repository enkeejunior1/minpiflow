# implementation of Rectified Flow for simple minded people like me.
import argparse
import torch.nn.functional as F
import torch.nn as nn
import torch
from einops import rearrange

class PiFlow:
    """Take linear schedule"""
    def __init__(self, student_model, teacher_model, ln=True, NFE=4):
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.ln = ln
        self.NFE = NFE

        self.in_channels = student_model.in_channels
        self.input_size = student_model.input_size

    def pi(self, x_t, t, *args, A_s=None, mu_s=None, sigma_s=None, x_s=None, s=None, **kwargs):
        assert len(A_s.shape) == len(mu_s.shape) == len(sigma_s.shape) == len(x_s.shape) == len(s.shape) == 5, f"A_s: {A_s.shape}, mu_s: {mu_s.shape}, sigma_s: {sigma_s.shape}, x_s: {x_s.shape}, s: {s.shape}"
        assert len(x_t.shape) == 4, f"x_t: {x_t.shape}"
        assert len(t.shape) == 1, f"t: {t.shape}"

        # convert into q(x_0 | x_s)
        x_t = x_t[:, None, :, :, :]
        t = t[:, None, None, None, None]
        mu_x = x_s - (1 - s) * mu_s
        sigma_x = s * sigma_s

        # convert into q(x_0 | x_t), Appendix F
        nu_x = s**2 * (1-t) * x_t - t**2 * (1-s) * x_s
        xi_x = s**2 * (1-t)**2 - t**2 * (1-s)**2

        a_t = A_s.log() - 1/2 * F.mse_loss(
            nu_x, xi_x*mu_x, reduction='none'
        ).mean(dim=[-2,-1], keepdim=True) / (xi_x*s**2*t + xi_x**2*sigma_x**2)

        A_t = a_t.softmax(dim=1)
        mu_t = (sigma_x**2 * nu_x + s**2 * t**2 * mu_x) / (sigma_x**2 * xi_x + s**2 * t**2)
        # sigma_t = ((sigma_x**2 * s**2 * t**2) / (sigma_x**2 * xi_x + s**2 * t**2)).sqrt()

        x_0 = (A_t * mu_t).sum(dim=1, keepdim=True)
        v_t = (x_t.squeeze(1) - x_0.squeeze(1)) / t.squeeze(1)
        return v_t

    @torch.no_grad()
    def from_s_to_t(self, x_s, s, t, cond, pi, NFE=50):
        dt = (s - t) / NFE
        assert (dt > 0).all(), f"dt < 0"
        for _ in range(NFE):
            v_s = pi(x_s, s, cond)
            x_s = x_s - dt[:, None, None, None] * v_s
            s = s - dt
        return x_s

    def forward_fm(self, z0, cond):
        b = z0.size(0)
        if self.ln:
            nt = torch.randn((b,)).to(z0.device)
            t = torch.sigmoid(nt)
        else:
            t = torch.rand((b,)).to(z0.device)
        texp = t.view([b, *([1] * len(z0.shape[1:]))])
        z1 = torch.randn_like(z0)
        zt = (1 - texp) * z0 + texp * z1
        vtheta = self.teacher_model(zt, t, cond)
        loss = F.mse_loss(vtheta, z1 - z0)
        return loss.mean()

    def forward_pi_data_dependent(self, z0=None, cond=None):
        """Algorithm 2
        s: start time
        t: end time
        """
        b = z0.size(0)
        s = torch.randint(1, self.NFE + 1, (b,)).to(z0.device) / self.NFE
        t = s - 1 / self.NFE * torch.rand((b,)).to(z0.device)
        sexp = s.view([b, *([1] * len(z0.shape[1:]))])
        z1 = torch.randn_like(z0)
        zs = (1 - sexp) * z0 + sexp * z1
        params = self.student_model(zs, s, cond)

        pi = lambda x_t, t, cond: self.pi(x_t, t, cond, **params)
        zt = self.from_s_to_t(zs, s, t, cond, pi)
        with torch.no_grad():
            vt = self.teacher_model(zt, t, cond).detach()
        
        vtheta = pi(zt.detach(), t, cond)
        loss = F.mse_loss(vt, vtheta)
        return loss.mean()

    def forward_pi_data_free(self, z0=None, cond=None):
        """Algorithm 3"""
        b = z0.size(0)
        t = torch.rand((b,)).to(z0.device)
        texp = t.view([b, *([1] * len(z0.shape[1:]))])
        z1 = torch.randn_like(z0)
        zt = (1 - texp) * z0 + texp * z1
        vtheta = self.teacher_model(zt, t, cond)
        loss = F.mse_loss(vtheta, z1 - z0)
        return loss.mean()

    @torch.no_grad()
    def sample_fm(self, x_t, cond, sample_steps=50):
        b = x_t.size(0)
        dt = 1.0 / sample_steps
        dt = torch.tensor([dt] * b).to(x_t.device).view([b, *([1] * len(x_t.shape[1:]))])
        images = [x_t]
        for i in range(sample_steps, 0, -1):
            t = i / sample_steps
            t = torch.tensor([t] * b).to(x_t.device)

            v_t = self.teacher_model(x_t, t, cond)
            x_t = x_t - dt * v_t
            images.append(x_t)
        return images

    @torch.no_grad()
    def sample_pi(self, x_s, cond):
        b = x_s.size(0)
        dt = 1 / self.NFE
        images = [x_s]
        
        s = torch.ones((b,)).to(x_s.device)
        for i in range(self.NFE, 0, -1):
            t = s - dt
            params = self.student_model(x_s, s, cond)
            pi = lambda x_t, t, cond: self.pi(x_t, t, s, cond, **params)
            x_s = self.from_s_to_t(x_s, s, t, cond, pi)
            images.append(x_s)
            s = t
        return images

if __name__ == "__main__":
    # train class conditional RF on mnist.
    import os
    import numpy as np
    import torch.optim as optim
    from PIL import Image
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    from torchvision.utils import make_grid
    from tqdm import tqdm

    import wandb
    from dit import DiT_Llama

    parser = argparse.ArgumentParser(description="use cifar?")
    parser.add_argument("--cifar", action="store_true")
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()
    CIFAR = args.cifar

    os.makedirs("weights", exist_ok=True)
    os.makedirs("contents", exist_ok=True)

    if CIFAR:
        dataset_name = "cifar"
        fdatasets = datasets.CIFAR10
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        channels = 3
        teacher_model = DiT_Llama(
            channels, 32, dim=256, n_layers=10, n_heads=8, num_classes=10
        ).cuda()
        student_model = DiT_Llama(
            channels, 32, dim=256, n_layers=10, n_heads=8, num_classes=10, K=8
        ).cuda()

    else:
        dataset_name = "mnist"
        fdatasets = datasets.MNIST
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Pad(2),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        channels = 1
        teacher_model = DiT_Llama(
            channels, 32, dim=64, n_layers=6, n_heads=4, num_classes=10
        ).cuda()
        student_model = DiT_Llama(
            channels, 32, dim=64, n_layers=6, n_heads=4, num_classes=10, K=8
        ).cuda()

    rf = PiFlow(student_model, teacher_model, NFE=4)
    train_ds = fdatasets(root="./data", train=True, download=True, transform=transform)
    train_dl = DataLoader(train_ds, batch_size=256, shuffle=True, drop_last=True)

    wandb.init(project=f"piflow_{dataset_name}")

    # train teacher model
    if os.path.exists(f"weights/teacher_{dataset_name}.pth") and not args.debug:
        rf.teacher_model.load_state_dict(torch.load(f"weights/teacher_{dataset_name}.pth"))
    else:
        epochs = 25 if not args.debug else 1
        optimizer = optim.Adam(rf.teacher_model.parameters(), lr=5e-4)
        for epoch in range(epochs):
            for i, (x, c) in tqdm(enumerate(train_dl)):
                x, c = x.cuda(), c.cuda()
                optimizer.zero_grad()
                loss = rf.forward_fm(x, c)
                loss.backward()
                optimizer.step()
                wandb.log({"teacher_loss": loss.item()})
                if args.debug:
                    break
        torch.save(rf.teacher_model.state_dict(), f"weights/teacher_{dataset_name}.pth")

    # train student model
    epochs = 100 if not args.debug else 1
    optimizer = optim.Adam(rf.student_model.parameters(), lr=5e-4)
    for epoch in range(epochs):
        for i, (x, c) in tqdm(enumerate(train_dl)):
            x, c = x.cuda(), c.cuda()
            optimizer.zero_grad()
            loss = rf.forward_pi_data_dependent(x, c)
            loss.backward()
            optimizer.step()
            wandb.log({"student_loss": loss.item()})
            
            if args.debug:
                break

        rf.student_model.eval()
        with torch.no_grad():
            cond = torch.arange(0, 16).cuda() % 10
            x_T = torch.randn(
                16, channels, 32, 32, generator=torch.Generator().manual_seed(42)
            ).cuda()
            
            # >>> student model with NFE = 4
            images = rf.sample_pi(x_T, cond)
            gif = []
            for image in images:
                # unnormalize
                image = image * 0.5 + 0.5
                image = image.clamp(0, 1)
                x_as_image = make_grid(image.float(), nrow=4)
                img = x_as_image.permute(1, 2, 0).cpu().numpy()
                img = (img * 255).astype(np.uint8)
                gif.append(Image.fromarray(img))

            gif[0].save(
                f"contents/sample_{epoch}_pi.gif",
                save_all=True,
                append_images=gif[1:],
                duration=100,
                loop=0,
            )

            last_img = gif[-1]
            last_img.save(f"contents/sample_{epoch}_pi_last.png")

            # >>> teacher model with NFE = 4
            images = rf.sample_fm(x_T, cond)
            gif = []
            for image in images:
                # unnormalize
                image = image * 0.5 + 0.5
                image = image.clamp(0, 1)
                x_as_image = make_grid(image.float(), nrow=4)
                img = x_as_image.permute(1, 2, 0).cpu().numpy()
                img = (img * 255).astype(np.uint8)
                gif.append(Image.fromarray(img))

            gif[0].save(
                f"contents/sample_{epoch}_fm.gif",
                save_all=True,
                append_images=gif[1:],
                duration=100,
                loop=0,
            )

            last_img = gif[-1]
            last_img.save(f"contents/sample_{epoch}_fm_last.png")

        rf.student_model.train()
        
        

