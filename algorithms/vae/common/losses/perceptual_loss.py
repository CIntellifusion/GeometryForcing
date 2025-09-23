import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .lpips import LPIPS
from .discriminator import (
    NLayerDiscriminator,
    NLayerDiscriminator3D,
    weights_init,
)


def adopt_weight(weight, global_step, threshold=0, value=0.0):
    if global_step < threshold:
        weight = value
    return weight


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1.0 - logits_real))
    loss_fake = torch.mean(F.relu(1.0 + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    # pylint: disable=not-callable
    d_loss = 0.5 * (
        torch.mean(F.softplus(-logits_real)) + torch.mean(F.softplus(logits_fake))
    )
    return d_loss


def l1(x, y):
    return torch.abs(x - y)


def l2(x, y):
    return torch.pow((x - y), 2)


class LPIPSWithDiscriminator(nn.Module):
    def __init__(
        self,
        disc_start,
        logvar_init=0.0,
        kl_weight=1.0,
        pixelloss_weight=1.0,
        disc_num_layers=3,
        disc_in_channels=3,
        disc_factor=1.0,
        disc_weight=1.0,
        perceptual_weight=1.0,
        use_actnorm=False,
        disc_conditional=False,
        disc_loss="hinge",
    ):

        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.kl_weight = kl_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight
        # output log variance
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)

        self.discriminator = NLayerDiscriminator(
            input_nc=disc_in_channels, n_layers=disc_num_layers, use_actnorm=use_actnorm
        ).apply(weights_init)
        self.discriminator_iter_start = disc_start
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(
                nll_loss, self.last_layer[0], retain_graph=True
            )[0]
            g_grads = torch.autograd.grad(
                g_loss, self.last_layer[0], retain_graph=True
            )[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(
        self,
        inputs,
        reconstructions,
        posteriors,
        optimizer_idx,
        global_step,
        last_layer=None,
        cond=None,
        split="train",
        weights=None,
    ):
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(
                inputs.contiguous(), reconstructions.contiguous()
            )
            rec_loss = rec_loss + self.perceptual_weight * p_loss

        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        weighted_nll_loss = nll_loss
        if weights is not None:
            weighted_nll_loss = weights * nll_loss
        weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
        nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        kl_loss = posteriors.kl()
        kl_loss = kl_loss.mean()

        # now the GAN part
        if optimizer_idx == 0:
            # generator update
            if cond is None:
                assert not self.disc_conditional
                logits_fake = self.discriminator(reconstructions.contiguous())
            else:
                assert self.disc_conditional
                logits_fake = self.discriminator(
                    torch.cat((reconstructions.contiguous(), cond), dim=1)
                )
            g_loss = -torch.mean(logits_fake)

            if self.disc_factor > 0.0:
                try:
                    d_weight = self.calculate_adaptive_weight(
                        nll_loss, g_loss, last_layer=last_layer
                    )
                except RuntimeError:
                    assert not self.training
                    d_weight = torch.tensor(0.0)
            else:
                d_weight = torch.tensor(0.0)

            disc_factor = adopt_weight(
                self.disc_factor, global_step, threshold=self.discriminator_iter_start
            )
            loss = (
                weighted_nll_loss
                + self.kl_weight * kl_loss
                + d_weight * disc_factor * g_loss
            )

            log = {
                "{}/total_loss".format(split): loss.clone().detach().mean().item(),
                # pylint: disable=not-callable
                "{}/logvar".format(split): self.logvar.detach().item(),
                "{}/kl_loss".format(split): kl_loss.detach().mean().item(),
                "{}/nll_loss".format(split): nll_loss.detach().mean().item(),
                "{}/rec_loss".format(split): rec_loss.detach().mean().item(),
                "{}/d_weight".format(split): d_weight.detach().item(),
                "{}/disc_factor".format(split): disc_factor,
                "{}/g_loss".format(split): g_loss.detach().mean().item(),
            }

            return loss, log

        if optimizer_idx == 1:
            # second pass for discriminator update
            if cond is None:
                logits_real = self.discriminator(inputs.contiguous().detach())
                logits_fake = self.discriminator(reconstructions.contiguous().detach())
            else:
                logits_real = self.discriminator(
                    torch.cat((inputs.contiguous().detach(), cond), dim=1)
                )
                logits_fake = self.discriminator(
                    torch.cat((reconstructions.contiguous().detach(), cond), dim=1)
                )

            disc_factor = adopt_weight(
                self.disc_factor, global_step, threshold=self.discriminator_iter_start
            )
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {
                "{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                "{}/logits_real".format(split): logits_real.detach().mean(),
                "{}/logits_fake".format(split): logits_fake.detach().mean(),
            }
            return d_loss, log


class LPIPSWithDiscriminator3D(nn.Module):
    def __init__(
        self,
        disc_start,
        kl_weight=1.0,
        perceptual_weight=1.0,
        # --- Discriminator Loss ---
        disc_num_layers=3,
        disc_in_channels=3,
        disc_factor=1.0,
        disc_weight=1.0,
        disc_use_actnorm=False,
        disc_loss="hinge",
        loss_type: str = "l1",
    ):

        super().__init__()
        self.perceptual_loss = LPIPS().eval()
        self.kl_weight = kl_weight
        self.perceptual_weight = perceptual_weight
        self.discriminator = NLayerDiscriminator3D(
            input_nc=disc_in_channels,
            n_layers=disc_num_layers,
            use_actnorm=disc_use_actnorm,
        ).apply(weights_init)
        self.disc_iter_start = disc_start
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
        self.disc_factor = disc_factor
        self.disc_weight = disc_weight
        self.loss_func = l1 if loss_type == "l1" else l2
        self.prev_d_weight = torch.tensor(0.0)

    def calculate_adaptive_weight(self, nll_loss, g_adversarial_loss, last_layer):
        nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(
            g_adversarial_loss, last_layer, retain_graph=True
        )[0]
        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        return d_weight * self.disc_weight

    def forward(
        self,
        inputs,
        reconstructions,
        posteriors,
        optimizer_idx,
        global_step,
        namespace="train",
        last_layer=None,
    ):
        is_training = reconstructions.requires_grad
        is_discriminator_iter_started = global_step >= self.disc_iter_start

        t = inputs.shape[2]
        # GAN Part
        if optimizer_idx == 0:
            inputs = rearrange(inputs, "b c t h w -> (b t) c h w").contiguous()
            reconstructions = rearrange(
                reconstructions, "b c t h w -> (b t) c h w"
            ).contiguous()
            rec_loss = self.loss_func(inputs, reconstructions)
            if self.perceptual_weight > 0:
                p_loss = self.perceptual_loss(inputs, reconstructions)
            else:
                p_loss = torch.zeros_like(rec_loss)
            nll_loss = (
                rec_loss + self.perceptual_weight * p_loss
            ).sum() / rec_loss.shape[0]
            kl_loss = posteriors.kl()
            kl_loss = (
                kl_loss.mean()
            )  # this ensures that the kl_loss is per batch & frame

            inputs = rearrange(inputs, "(b t) c h w -> b c t h w", t=t).contiguous()
            reconstructions = rearrange(
                reconstructions, "(b t) c h w -> b c t h w", t=t
            ).contiguous()

            logits_fake = self.discriminator(reconstructions)
            g_adversarial_loss = -torch.mean(logits_fake)
            if is_training:
                if is_discriminator_iter_started:
                    if self.disc_factor > 0.0:
                        d_weight = self.calculate_adaptive_weight(
                            nll_loss, g_adversarial_loss, last_layer=last_layer
                        )
                    else:
                        d_weight = torch.tensor(1.0)
                else:
                    d_weight = torch.tensor(0.0)
                self.prev_d_weight = d_weight.clone().detach()
            else:
                d_weight = self.prev_d_weight
            if not is_discriminator_iter_started:
                g_adversarial_loss = torch.tensor(0.0, requires_grad=is_training)

            g_adversarial_loss_weight = (
                adopt_weight(
                    self.disc_factor, global_step, threshold=self.disc_iter_start
                )
                * d_weight
            )
            g_loss = nll_loss + self.kl_weight * kl_loss + d_weight * g_adversarial_loss
            g_log = {
                f"{namespace}/g_loss": g_loss.clone().detach(),
                f"{namespace}/reconstruction_loss": rec_loss.detach().mean(),
                f"{namespace}/perceptual_loss": p_loss.detach().mean(),
                f"{namespace}/kl_loss": kl_loss.detach(),
                f"{namespace}/g_adversarial_loss": g_adversarial_loss.detach(),
                f"{namespace}/g_adversarial_loss_weight": g_adversarial_loss_weight.detach(),
            }
            return g_loss, g_log
        elif optimizer_idx == 1:
            logits_real = self.discriminator(inputs.contiguous().detach())
            logits_fake = self.discriminator(reconstructions.contiguous().detach())

            disc_factor = adopt_weight(
                self.disc_factor, global_step, threshold=self.disc_iter_start
            )

            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            d_log = {
                f"{namespace}/disc_loss": d_loss.clone().detach().mean(),
                f"{namespace}/logits_real": logits_real.detach().mean(),
                f"{namespace}/logits_fake": logits_fake.detach().mean(),
            }
            return d_loss, d_log
