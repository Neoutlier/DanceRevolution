import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.special import comb as n_over_k



# DM: the fitting algorithm was adapted from https://stackoverflow.com/a/62225617


class BezierFitter:
    def __init__(self):
        self.points = {}
        self.bernstein_matrices = {}
        self.overlap_t_points = {}
        self.windows = {}

    @staticmethod
    def torch_binom(n, k):
        # # from https://github.com/pytorch/pytorch/issues/47841 and
        # # https://discuss.pytorch.org/t/n-choose-k-function/121974/2
        mask = n.detach() >= k.detach()
        n = mask * n
        k = mask * k
        a = torch.lgamma(n + 1) - torch.lgamma((n - k) + 1) - torch.lgamma(k + 1)
        return torch.exp(a) * mask

        # return torch.prod(torch.tensor([(n + 1 - i)/i for i in torch.arange(start=1, end=k+1)]))

    @staticmethod
    def bernstein_poly(n, t, k, pytorch=False, device=None):
        """ Bernstein polynomial when a = 0 and b = 1. """
        if pytorch:
            # nok = torch.prod(torch.tensor([(n + 1 - i)/i for i in torch.arange(start=1, end=k+1)]))
            nok = BezierFitter.torch_binom(n, k)
        else:
            nok = n_over_k(n, k)

        return t ** k * (1 - t) ** (n - k) * nok

    @staticmethod
    def bernstein_matrix(t, degree, pytorch=False, dtype_np=np.float32, dtype_torch=torch.float32, device=None,
                         swap=False):
        """ Bernstein matrix for Bézier curves. """
        if pytorch:
            assert not swap
            return torch.tensor([[BezierFitter.bernstein_poly(degree, t, k, pytorch=True, device=device)
                                  for k in torch.arange(start=0, end=degree + 1, device=device, dtype=torch.float32)]
                                 for t in t], dtype=dtype_torch, device=device)
        else:
            if swap:
                return np.array([[BezierFitter.bernstein_poly(degree, t, k, pytorch=False) for t in t]
                                 for k in range(degree + 1)]).astype(dtype_np)
            else:
                return np.array([[BezierFitter.bernstein_poly(degree, t, k, pytorch=False)
                                  for k in range(degree + 1)] for t in t], dtype=dtype_np)

    @staticmethod
    def least_square_fit(points, m, pytorch=False, out=None):
        if pytorch:
            m_ = torch.pinverse(m)

            if not isinstance(points, torch.Tensor):
                points = torch.tensor(points, device=m_.device)

            return m_ @ points
        else:
            m_ = np.linalg.pinv(m)
            return np.matmul(m_, points, out=out)

    @staticmethod
    def casteljau(control_points, t, return_all_points=False):
        n = control_points.shape[0]
        q = np.array(control_points)
        r = []

        for k in range(1, n):
            for i in range(0, n - k):
                q[i] = (1 - t) * q[i] + t * q[i + 1]

            qq = np.array(q[:-k])

            if return_all_points:
                r.append(qq)
            else:
                r = qq

        return r

    @staticmethod
    def casteljau_torch(control_points, t, return_all_points=False):
        n = control_points.shape[-2]  # De Casteljau's algorithm
        q = control_points.clone()  # yes, with grad history
        r = []

        for k in range(1, n):
            for i in range(0, n - k):
                q[..., i, :] = (1 - t) * q[..., i, :] + t * q[..., i + 1, :]

            qq = q[..., :-k, :]  # I guess we don't need clone here for autograd

            if return_all_points:
                r.append(qq)
            else:
                r = qq

        return r

    @staticmethod
    def generate_bezier_curve_with_casteljau(control_points, t_points, pytorch=False):
        if pytorch:
            return torch.cat([BezierFitter.casteljau_torch(control_points, t, return_all_points=False)
                              for t in t_points])
        else:
            return np.concatenate([BezierFitter.casteljau(control_points, t, return_all_points=False)
                                   for t in t_points])

    @staticmethod
    def get_intermediate_points_series(control_points, t, points_axis=0, cp_axis=2, cython=True, zero_if_no_cp=True):
        """
        Expects array of shape (nodes, channels, order+1)
        Returns array of shape (nodes, get_n_intermediate_points * channels)
        """
        assert points_axis == 0 and cp_axis == 2 and control_points.ndim == 3, \
            'This function expects control_points to be in shape (nodes, channels, order+1)'
        assert cython, 'Use cython mode'

        if not control_points.any():
            if zero_if_no_cp:
                nodes, channels, order_p1 = control_points.shape
                return np.zeros(nodes, channels * BezierFitter.get_n_intermediate_points(order_p1 - 1))
            else:
                raise RuntimeError('Empty control points')

        l = control_points.shape[points_axis]

        if cython:
            try:
                from casteljau import casteljau_all_nodes
                return casteljau_all_nodes(control_points, t)
            except ImportError:
                print('Compile cython module to run Casteljau')
                return
        else:
            return np.stack([np.concatenate(BezierFitter.casteljau(control_points[i, :, :].T, t))
                             for i in range(l)])

    @staticmethod
    def get_n_intermediate_points(order):
        n = order + 1
        i = 0

        for k in range(1, n - 1):
            i += n - k

        return i

    def get_bernstein_matrix(self, t, degree, swap=False, device=None, pytorch=False):
        # this creates a bezier curve with the given degree
        k = f'{degree}_{len(t)}_swap={swap}_pytorch={pytorch}'

        if k in self.bernstein_matrices:
            m = self.bernstein_matrices[k]
        else:
            m = BezierFitter.bernstein_matrix(t, degree, pytorch=pytorch, device=device, swap=swap)
            self.bernstein_matrices[k] = m

        return m

    def get_points(self, n_points, device=None, pytorch=False):
        k = f'{n_points}_pytorch={pytorch}'

        if k in self.points:
            t = self.points[k]
        else:
            t = torch.linspace(0, 1, n_points, device=device) if pytorch else np.linspace(0, 1, n_points)
            self.points[k] = t

        return t

    def get_windows(self, sliding_w, w_overlap, trajectory_length, pytorch=False, as_list=True):
        key = f'{sliding_w}-{w_overlap}-{trajectory_length}-pytorch={pytorch}'

        if key in self.windows:
            return self.windows[key]

        if trajectory_length > sliding_w:
            windows = torch.arange(0, trajectory_length).unfold(0, sliding_w, sliding_w - w_overlap)
        else:
            windows = torch.arange(0, trajectory_length).unsqueeze(0)

        if not pytorch:
            windows = windows.numpy()

        if as_list:
            windows = windows.tolist()

        self.windows[key] = windows

        return windows

    def get_overlap_t_points(self, overlap, pytorch=False):
        key = f'{overlap}-pytorch={pytorch}'

        if key in self.overlap_t_points:
            return self.overlap_t_points[key]

        if pytorch:
            p = torch.linspace(0, 1, overlap)[None, None, :]
        else:
            p = np.linspace(0, 1, overlap)[None, None, :]

        self.overlap_t_points[key] = p

        return p

    def fit_bezier(self, points, degree=3, pytorch=False, device=None, out=None):
        """ Least square Bezier fit using penrose pseudo-inverse.
        Parameters:
        points: a nxd array o n d-dimensional points
        degree: degree of the Bézier curve. 2 for quadratic, 3 for cubic.
        Based on https://stackoverflow.com/questions/12643079/b%C3%A9zier-curve-fitting-with-scipy
        and probably on the 1998 thesis by Tim Andrew Pastva, "Bézier Curve Fitting".
        """
        n_points = len(points)
        assert degree >= 1, 'degree must be greater.'
        assert n_points >= degree + 1, f'There must be at least {degree + 1} points to determine the parameters of a ' \
                                       f'degree {degree} curve. Got only {n_points} points.'

        t = self.get_points(n_points, device, pytorch)
        m = self.get_bernstein_matrix(t, degree, device, pytorch)

        # we minimise the error between the given points and the curve
        control_points = BezierFitter.least_square_fit(points, m, pytorch=pytorch, out=out)

        return control_points

    def fit_bezier_series(self, points_matrix, degree=3, time_axis=1, nodes_axis=2, dtype=np.float32,
                          zero_if_not_enough_points=True):
        """
        Expects array of shape (channels, frames, nodes)
        returns array of shape (nodes, channels, degree+1)
        """
        n_points = points_matrix.shape[time_axis]
        assert degree >= 1, 'degree must be greater than 1'
        assert time_axis == 1 and nodes_axis == 2 and points_matrix.ndim == 3, \
            'This function expects points_matrix to be in shape (channels, frames, nodes)'

        if n_points < degree + 1:
            if zero_if_not_enough_points:
                return np.zeros((points_matrix.shape[2], points_matrix.shape[0], degree+1), dtype=dtype)
            else:
                raise RuntimeError(f'There must be at least {degree + 1} points to determine the parameters of a '
                                   f'degree {degree} curve. Got only {n_points} points.')

        t = self.get_points(n_points)

        # note that m is swapped in this function wrt to the other m elsewhere
        m = self.get_bernstein_matrix(t, degree, swap=True)

        m_ = np.linalg.pinv(m).astype(dtype)
        cp = np.array([np.matmul(points_matrix[:, :, i], m_) for i in range(points_matrix.shape[nodes_axis])])

        return cp

    def fit_bezier_series_with_windows(self, points_matrix, degree, window, overlap, time_axis=1, nodes_axis=2,
                                       inter_points=None, target_length=None, out_window=None, out_thr=0.1,
                                       save_idx=None, frames_list=None, bounds=None,
                                       not_enough_points_policy='err'):
        """
        Expects array of shape (channels, frames, nodes)
        This function returns both the control points and the joint curve. See other functions for the output shapes
        """
        trajectory_length = points_matrix.shape[time_axis]
        assert trajectory_length == len(frames_list), f'Trajectory length {target_length} did not match frames list ' \
                                                      f'length {len(frames_list)}'
        assert 0 <= overlap <= 1 or overlap % 2 == 0, 'Overlap must be either 0, 1 or multiple of 2'

        assert degree >= 1, 'degree must be greater than 1'
        assert time_axis == 1 and nodes_axis == 2 and points_matrix.ndim == 3, \
            'This function expects points_matrix to be in shape (channels, frames, nodes)'

        assert target_length is None and inter_points is None and frames_list is not None and bounds is not None, \
            'Review this to work with target length or interpolation points'

        if trajectory_length < degree + 1:
            if not_enough_points_policy == 'warn':
                warnings.warn(f'There must be at least {degree + 1} points to determine the parameters of a '
                              f'degree {degree} curve. Got only {trajectory_length} points. '
                              f'Returning the input sequence')
                return points_matrix, [], [], []
            else:
                raise RuntimeError(f'There must be at least {degree + 1} points to determine the parameters of a '
                                   f'degree {degree} curve. Got only {trajectory_length} points.')

        windows = self.get_windows(window, overlap, trajectory_length)
        t_overlap = self.get_overlap_t_points(overlap) if overlap > 0 else 0
        n_windows = len(windows)
        cps = []
        gs = []
        outliers_all = []

        # last_window_points = None
        save_idx = [] if save_idx is None else save_idx

        for i, idx in enumerate(windows):
            if i == n_windows - 1 and idx[-1] != trajectory_length - 1:
                idx = np.arange(idx[0], trajectory_length, 1)
                # last_window_points = target_length - (i * inter_points)

            if out_window is not None:
                outliers = find_outliers(points_matrix, idx, save_idx, out_window, outlier_thr=out_thr)
                idx = [k for k in idx if k not in outliers]

                for o in outliers:
                    if o not in outliers_all:
                        outliers_all.append(o)

            points_in_windows = points_matrix[:, idx, :]
            cp = self.fit_bezier_series(points_in_windows, degree=degree, time_axis=time_axis, nodes_axis=nodes_axis)
            cps.append(cp)

            if i == 0:
                n_points = frames_list[idx[-1]] - bounds[0] + 1
            elif i < n_windows - 1:
                n_points = frames_list[idx[-1]] - frames_list[idx[0]] + 1
            else:
                n_points = bounds[1] - frames_list[idx[0]] + 1

            # n_points += overlap  # don't do this or the sequence will be misaligned with the video

            # n_points = last_window_points if i == n_windows - 1 and last_window_points is not None else inter_points
            g = self.get_bezier_curves(cp, degree=degree, inter_points=n_points)
            gs.append(g)

            i += 1

        n_splits = len(gs)
        segs_o = []

        if n_splits == 1:
            segs_o = [gs[0]]
        else:
            for i in range(n_splits - 1):
                a = gs[i]  # we are concatenating the unique bits of a and the midpoint in the overlapping area
                b = gs[i + 1]

                start_a = overlap if i > 0 else 0
                # end_a = window - overlap
                end_a = -overlap
                start_b = overlap
                aa = a[:, :, start_a:end_a]

                if overlap > 0:
                    oo = (1 - t_overlap) * a[:, :, end_a:] + t_overlap * b[:, :, :start_b]
                    segs_o.extend([aa, oo])
                else:
                    segs_o.extend([aa])

                if i == n_splits - 2:
                    bb = b[:, :, start_b:]  # finally, we add the last segment
                    segs_o.append(bb)

        final_curve = np.concatenate(segs_o, axis=2)
        expected_length = bounds[1] - bounds[0] + 1
        resampling = np.linspace(0, final_curve.shape[2] - 1, expected_length, dtype=np.int32)
        final_curve = final_curve[:, :, resampling]

        outliers_all = sorted(outliers_all)

        return final_curve, gs, cps, outliers_all

    def get_bezier_curves(self, control_points, degree, inter_points=100, points_axis=0, dtype=np.float32,
                          zero_if_no_cp=True):
        """
        Expects array of shape (nodes, channels, degree+1)
        Returns array of shape (nodes, channels, inter_points)
        """
        assert degree >= 1, 'degree must be greater than 1'
        assert points_axis == 0 and control_points.ndim == 3, \
            'This function expects control_points to be in shape (nodes, channels, order+1)'

        if not control_points.any():
            if zero_if_no_cp:
                # empty control points, useful to concatenate stuff from different bodies
                return np.zeros((control_points.shape[0], control_points.shape[1], inter_points), dtype=dtype)
            else:
                raise RuntimeError('Empty control points')

        t = self.get_points(inter_points)

        # note that m is swapped in this function wrt to the other m elsewhere
        m = self.get_bernstein_matrix(t, degree, swap=True)

        b = np.array([np.matmul(control_points[i, :, :], m) for i in range(control_points.shape[points_axis])])
        return b

    def get_bezier_curve(self, control_points, degree, inter_points=100, pytorch=False, device=None, out=None,
                         swap=False):
        assert len(control_points) == degree + 1, 'Need degree + 1 control points'
        t = self.get_points(inter_points)
        m = self.get_bernstein_matrix(t, degree, swap=swap)

        if pytorch:
            return m @ control_points
        else:
            return np.matmul(m, control_points, out=out)