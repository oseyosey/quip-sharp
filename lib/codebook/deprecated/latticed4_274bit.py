"""
builds a deep-hole-centered D4 codebook
this is a codebook consisting of points on the lattice in R4
    where each component is a half-integer
    and the components sum to an even number
from this lattice, we select the points that have a norm-squared of at most 9
this results in a codebook of 256 points distributed as follows
    8 with sorted abs of [1/2, 1/2, 1/2, 1/2]
    8                    [3/2, 3/2, 3/2, 3/2]
    4c2 * 8 = 48         [1/2, 1/2. 3/2, 3/2]
    4 * 8 = 32           [1/2, 1/2, 1/2, 3/2]
    4 * 8 = 32           [1/2, 3/2, 3/2, 3/2]
    4 * 8 = 32           [1/2, 1/2, 1/2, 5/2]
    4 * 3 * 8 = 96       [1/2, 1/2, 3/2, 5/2]
"""

import torch
from torch import nn
import quiptools_cuda

from lib.utils.matmul_had import matmul_hadU_cuda, matmul_hadUt_cuda

_D4274B_CODESZ = 4


def get_grid():
    hintr = torch.arange(-14, 14) + 1/2
    gd4 = torch.cartesian_prod(*[hintr] * _D4274B_CODESZ)
    gd4m2 = (gd4.sum(dim=-1) % 2 == 0)
    gd4n = gd4.norm(dim=-1)**2 <= 27
    d4 = gd4[torch.where(gd4m2 * gd4n)[0]]
    return d4


_D4274B_CACHED = get_grid()
_D4274B_NORM_CACHED = torch.diag(_D4274B_CACHED @ _D4274B_CACHED.T)


class D4274B_codebook(nn.Module):

    def __init__(self, build_truncated=True):
        super(D4274B_codebook, self).__init__()
        self.opt_scale = 1.83
        self.codesz = _D4274B_CODESZ
        self.idx_dtype = torch.int16
        self.idx_offset = -2**15

        grid = _D4274B_CACHED
        self.register_buffer('grid', grid)
        self.register_buffer('grid_norm', _D4274B_NORM_CACHED)

        if build_truncated:
            idxs = torch.where(
                ((grid[:, 1:] < 0).sum(dim=-1) <= 1) * \
                (grid[:, 1:].min(dim=-1).values >= -0.5)
            )[0]
            grid_part = grid[idxs]
            self.register_buffer('grid_part', grid_part)
            self.register_buffer('grid_part_norm', torch.diag(grid_part @ grid_part.T))
            self.register_buffer('int_map', 2**torch.arange(_D4274B_CODESZ))
            allcombo_idx, idx_map = self.iterate_mask()
            self.register_buffer('allcombo_idx', allcombo_idx)
            self.register_buffer('idx_map', idx_map)


        '''
        import numpy as np
        assert np.abs(np.log2(self.grid.shape[0])/4 - 2.74) < 1e-2
        self.cuda()
        samples = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(4), torch.eye(4)).rsample([200000]).cuda()
        print(samples.shape)
        def fn_s(s):
            err = (self.quantize(samples*s, False)/s - samples).float().norm()**2
            err = err.cpu() / torch.numel(samples)
            return err.cpu()        
        import scipy
        print(scipy.optimize.minimize_scalar(fn_s, bounds=(0.1, 100)))
        # exit()
        '''
            

    def to_int(self, mask):
        return (self.int_map.unsqueeze(0) * mask.int()).sum(dim=-1)

    def to_mask(self, int):
        return ((self.int_map & int.unsqueeze(-1)) > 0) * 2 - 1

    def round(self, X, grid, grid_norm):
        assert X.shape[-1] == self.codesz
        Xqidx = (2 * X @ grid.T - grid_norm).argmax(-1)
        return grid[Xqidx], Xqidx

    def iterate_mask(self, device=0):
        bmask = 2**torch.arange(self.codesz)
        flips = torch.stack([((torch.tensor([i]) & bmask) > 0).int()
                             for i in range(2**self.codesz)]).to(device)
        raw_idx = torch.where(flips.sum(dim=-1) % 2 == 0)[0]
        flips = 1 - 2 * flips[raw_idx]
        idx_map = torch.zeros(2**self.codesz, dtype=torch.int32)
        for i in range(len(raw_idx)):
            idx_map[raw_idx[i]] = i
        allcombo = flips.unsqueeze(1) * self.grid_part.unsqueeze(0).to(device)
        allcombo_idx = torch.zeros(allcombo.shape[0:2], dtype=self.idx_dtype)
        for i in range(len(allcombo)):
            allcombo_idx[i] = (
                self.round(allcombo[i], self.grid.to(device), self.grid_norm.to(device))[1] +
                self.idx_offset).to(self.idx_dtype)
        return allcombo_idx.cpu(), idx_map.cpu()

    def quantize(self, X, return_idx=True):
        
        X_part = torch.abs(X)
        X_odd = torch.where((X < 0).sum(dim=-1) % 2 != 0)[0]
        X_part[X_odd, 0] = -X_part[X_odd, 0]
        mask = 1 - 2 * (X < 0).to(torch.float32)
        mask[X_odd, 0] = -mask[X_odd, 0]
        roundout, Xqidx = self.round(X_part, self.grid_part, self.grid_part_norm)
        vals = roundout * mask

        if not return_idx:
            return vals

        real_idx = self.allcombo_idx[self.idx_map[self.to_int((1 - mask) / 2)], Xqidx]
        
        return vals, real_idx

    def by_idxs(self, idxs):
        return self.grid[idxs.int() - self.idx_offset]

    
class QuantizedD4274BLinear(nn.Module):

    def __init__(self, device):
        super().__init__()
        self.codebook = D4274B_codebook(build_truncated=False).to(device)
        self.codebook.grid = self.codebook.grid.to(torch.float16)

    def forward(self,
                input,
                Qidxs,
                SU,
                SV,
                Wscale,
                rank=-1,
                A=None,
                B=None,
                rescale_WH=False,
                scaleWH=None):
        (m, n) = Qidxs.shape

        x = input.view(-1, n * _D4274B_CODESZ).to(torch.float32)
        if rescale_WH:
            x /= scaleWH
        x = x * SU
        x = matmul_hadUt_cuda(x)

        if rank > 0:
            Bx = x @ B.t().to(torch.float32)
            ABx = Bx @ A.t().to(torch.float32)

        num_scale = 1024
        x = x / num_scale
        x = x.to(torch.float16)

        W_decompressed = self.codebook.by_idxs(Qidxs).reshape(-1, n * _D4274B_CODESZ)
        z = x @ W_decompressed.t()

        x = z.to(torch.float32)
        x = x * (Wscale * num_scale)

        if rank > 0:
            x = x + ABx.to(torch.float32)

        x = matmul_hadU_cuda(x)
        x = x * SV

        output = x.view(*input.shape[:-1], m)

        return output
