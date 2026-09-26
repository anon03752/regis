"""Compute the R3GAN gradient penalty with reverse-over-forward autodiff.

For P = gamma/2 * mean(||grad_x D||^2), set v = stopgrad(grad_x D).
The surrogate S = gamma * mean(JVP(D, v)) has the same parameter gradient
as P, while avoiding a double backward through grouped convolutions.
Its scalar value is 2P. Use the returned detached gradient norms to log
the conventional penalty value."""
import torch
import torch.autograd.forward_ad as fwAD

from models.train.r3gan import relativistic


def discriminator_loss(D, real, fake, t, gamma):
    """Returns (loss, adv, gp_real, gp_fake).
    real / fake: (B,·,H,W) tensors (no grad required); t: (B,) target classes.
    loss has the same parameter gradient as adv + 0.5*gamma*(gp_real.mean()+gp_fake.mean())
    computed with create_graph=True; gp_real and gp_fake are detached."""
    B = real.shape[0]
    x = torch.cat([real.detach(), fake.detach()], 0).requires_grad_(True)
    tt = torch.cat([t, t], 0)
    out = D(x, tt)                                   # (2B,)  one batched forward (D is per-sample)
    d_real, d_fake = out[:B], out[B:]
    adv = relativistic(d_real, d_fake)
    (g,) = torch.autograd.grad(out.sum(), x, retain_graph=True)          # ∇ₓD per sample, no create_graph
    gp = g.flatten(1).pow(2).sum(1)                                       # (2B,) = ||∇ₓD_b||^2
    gp_real, gp_fake = gp[:B].detach(), gp[B:].detach()
    with fwAD.dual_level():
        xd = fwAD.make_dual(x.detach(), g.detach())
        outd = D(xd, tt)
        jv = fwAD.unpack_dual(outd).tangent                               # (2B,) = <∇ₓD_b, g_b>
    surrogate = gamma * (jv[:B].mean() + jv[B:].mean())                   # dS/dθ = dP/dθ ; S = 2P
    return adv + surrogate, adv, gp_real, gp_fake
