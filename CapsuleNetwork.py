import torch
from torch import nn

# Capsule Layer
class CapsuleLayer(nn.Module):
    def __init__(self, in_units, in_channels, num_capsules, dim_capsules, num_routing=3, device="cuda"):
        super(CapsuleLayer, self).__init__()

        self.in_units = in_units
        self.in_channels = in_channels
        self.num_capsules = num_capsules
        self.dim_capsules = dim_capsules
        self.num_routing = num_routing
        self.device = device

        self.W = nn.Parameter(torch.randn(1, num_capsules, in_units, dim_capsules, in_channels)).to(device)

    def forward(self, x):
        batch_size = x.size(0)
        x = torch.stack([x]*self.num_capsules, dim=1).unsqueeze(4)
        W = self.W.unsqueeze(0)  # add an extra dimension for the batch size
        u_hat = torch.einsum('ijkl,ijkm->ijkm', W, x)


        # Initialise routing logits to zero.
        b_ij = torch.zeros(batch_size, self.num_capsules, self.in_units, 1).to(self.device)

        # Iterative routing.
        for iteration in range(self.num_routing - 1):
            c_ij = torch.softmax(b_ij, dim=1)
            s_j = (c_ij * u_hat).sum(dim=2, keepdim=True)
            v_j = self.squash(s_j)
            delta_b_ij = (u_hat * v_j).sum(dim=-1, keepdim=True)
            b_ij = b_ij + delta_b_ij

        c_ij = torch.softmax(b_ij, dim=1)
        s_j = (c_ij * u_hat).sum(dim=2, keepdim=True)
        v_j = self.squash(s_j)

        return v_j.squeeze(3)

    # Squashing function corresponding to Eq. 1
    def squash(self, s_j):
        s_j_norm = torch.norm(s_j, dim=-1, keepdim=True)
        return (s_j_norm / (1. + s_j_norm ** 2)) * s_j

# Full Capsule Network
class CapsuleNetwork(nn.Module):
    def __init__(self, img_shape, channels, primary_dim, num_classes, out_dim, num_routing):
        super(CapsuleNetwork, self).__init__()
        self.img_shape = img_shape
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(img_shape[0], channels, kernel_size=9, stride=1)
        self.relu = nn.ReLU(inplace=True)

        self.primary_capsules = CapsuleLayer(in_units=channels * 20 * 20, in_channels=channels, num_capsules=8, dim_capsules=primary_dim, num_routing=1)
        
        self.digit_capsules = CapsuleLayer(in_units=8, in_channels=primary_dim, num_capsules=num_classes, dim_capsules=out_dim, num_routing=num_routing)

    def forward(self, x):
        x = self.conv1(x.view(-1, *self.img_shape))  # adjust the size of the input to match the expected number of channels
        x = self.relu(x)
        x = self.primary_capsules(x)
        x = self.digit_capsules(x)
        return x


    def loss(self, images, labels, outputs):
        return self.margin_loss(outputs, labels)

    # Margin loss for Eq. 4, when digit of interest.
    def margin_loss(self, v_c, target):
        batch_size = target.size(0)
        v_c_norm = torch.norm(v_c, dim=2, keepdim=False)

        # Calculate left and right max() terms.
        zero = torch.zeros(1).to(self.device)
        m_plus = 0.9
        m_minus = 0.1
        max_l = torch.max(m_plus - v_c_norm, zero).view(batch_size, -1) ** 2
        max_r = torch.max(v_c_norm - m_minus, zero).view(batch_size, -1) ** 2

        # Calculate T_c: Eq. 2.
        T_c = torch.sparse.torch.eye(self.num_classes).to(self.device).index_select(dim=0, index=target)

        # Calculate L_c = T_c * max_l + lambda * (1 - T_c) * max_r.
        L_c = T_c * max_l + 0.5 * (1. - T_c) * max_r

        # Sum over the digit capsules.
        L_c = L_c.sum(dim=1)

        return L_c.mean()
