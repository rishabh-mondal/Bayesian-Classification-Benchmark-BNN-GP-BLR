import streamlit as st
import pandas as pd
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import hamiltorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from torchmetrics.classification import MulticlassCalibrationError
from sklearn.metrics import accuracy_score
import GPy


def generate_make_moons(seed):
    X, y = make_moons(n_samples=1000, noise=0.3, random_state=seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )
    return X_train, X_test, y_train, y_test


st.title("Make Moons Dataset Visualization with Different Seed")

seed = st.slider("Select Seed:", min_value=0, max_value=100, value=43)

X_train, X_test, y_train, y_test = generate_make_moons(seed)


st.subheader("Visualization:")
fig, ax = plt.subplots()
scatter = ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, alpha=0.7)
ax.set_title("Make Moons Dataset")
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
ax.legend(*scatter.legend_elements(), loc="lower right", title="Classes")
st.pyplot(fig)

X_train = torch.tensor(X_train, dtype=torch.float)
y_train = torch.tensor(y_train, dtype=torch.float).reshape((-1, 1))
X_test = torch.tensor(X_test, dtype=torch.float)
y_test = torch.tensor(y_test, dtype=torch.float).reshape((-1, 1))


# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.fc1 = nn.Linear(2, 100)
#         self.fc2 = nn.Linear(100, 100)
#         self.fc3 = nn.Linear(100, 2)

#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         #         x = F.softmax(self.fc3(x))
#         return x


# net = Net()

# tau_list = []
# tau = 10.0  # ./100. # 1/50
# for w in net.parameters():
#     tau_list.append(tau)
# tau_list = torch.tensor(tau_list)

# hamiltorch.set_random_seed(seed)

# params_init = hamiltorch.util.flatten(net)
# print(params_init.shape)

# step_size = 0.001
# num_samples = 300
# L = 20
# tau_out = 1.0
# normalizing_const = 1.0
# burn = 100

# params_hmc = hamiltorch.sample_model(
#     net,
#     X_train,
#     y_train,
#     params_init=params_init,
#     model_loss="multi_class_linear_output",
#     num_samples=num_samples,
#     burn=burn,
#     step_size=step_size,
#     num_steps_per_sample=L,
#     tau_out=tau_out,
#     tau_list=tau_list,
#     normalizing_const=normalizing_const,
# )


# pred_list, log_prob_list = hamiltorch.predict_model(
#     net,
#     x=X_test,
#     y=y_test,
#     samples=params_hmc,
#     model_loss="multi_class_log_softmax_output",
#     tau_out=1.0,
#     tau_list=tau_list,
# )
# _, pred = torch.max(pred_list, 2)
# acc = []
# acc = torch.zeros(int(len(params_hmc)) - 1)
# nll = torch.zeros(int(len(params_hmc)) - 1)
# ensemble_proba = F.softmax(pred_list[0], dim=-1)
# mcce = MulticlassCalibrationError(num_classes=2, n_bins=2)  # , norm='l2')

# for s in range(1, len(params_hmc)):
#     _, pred = torch.max(pred_list[:s].mean(0), -1)
#     acc[s - 1] = (pred.float() == y_test.flatten()).sum().float() / y_test.shape[0]
#     ensemble_proba += F.softmax(pred_list[s], dim=-1)
#     #     print(ensemble_proba.cpu()/(s+1),y_test[:].long().cpu().flatten())

#     out_calibration_error = mcce(
#         ensemble_proba.cpu() / (s + 1), y_test[:].long().cpu().flatten()
#     )
#     print(out_calibration_error)
#     nll[s - 1] = F.nll_loss(
#         torch.log(ensemble_proba.cpu() / (s + 1)),
#         y_test[:].long().cpu().flatten(),
#         reduction="mean",
#     )


# # Create separate plots for accuracy and negative log likelihood
# fig, (ax_acc, ax_nll) = plt.subplots(1, 2, figsize=(12, 5))

# # Plot Accuracy
# ax_acc.plot(acc)
# ax_acc.set_title("Accuracy")
# ax_acc.set_xlabel("Number of Samples")
# ax_acc.set_ylabel("Accuracy")

# # Plot Negative Log Likelihood
# ax_nll.plot(nll)
# ax_nll.set_title("Negative Log Likelihood")
# ax_nll.set_xlabel("Number of Samples")
# ax_nll.set_ylabel("NLL")

# fig.tight_layout()
# st.subheader("Model Evaluation:")
# st.pyplot(fig)

# st.subheader("BNN: mean and varience plot of the predictions:")
# posterior_samples = params_hmc  # .detach()
# # Consider burning the first 100 samples
# posterior_samples = posterior_samples  # [1000:]
# y_preds = []
# n_grid = 200
# lims = 4
# twod_grid = torch.tensor(
#     np.meshgrid(np.linspace(-lims, lims, n_grid), np.linspace(-lims, lims, n_grid))
# ).float()
# with torch.no_grad():
#     for theta in posterior_samples:
#         params_list = hamiltorch.util.unflatten(net, theta)
#         params = net.state_dict()
#         for i, (name, _) in enumerate(params.items()):
#             params[name] = params_list[i]
#         y_pred = torch.func.functional_call(
#             net, params, twod_grid.view(2, -1).T
#         ).squeeze()

#         y_preds.append(y_pred[:, 0])

# logits = torch.stack(y_preds).mean(axis=0).reshape(n_grid, n_grid)
# probs = torch.sigmoid(logits)


# fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# # Plot the first subplot
# axs[0].contourf(
#     twod_grid[0].cpu().numpy(),
#     twod_grid[1].cpu().numpy(),
#     probs.cpu().numpy(),
#     cmap="bwr",
#     alpha=0.5,
# )
# axs[0].scatter(
#     X_test[:, 0].cpu().numpy(),
#     X_test[:, 1].cpu().numpy(),
#     c=y_test.cpu().numpy(),
#     cmap="bwr",
#     alpha=0.5,
# )
# axs[0].set_title("Figure 1")

# # Plot the second subplot
# axs[1].contourf(
#     twod_grid[0].cpu().numpy(),
#     twod_grid[1].cpu().numpy(),
#     torch.stack(y_preds).std(axis=0).reshape(n_grid, n_grid).cpu().numpy(),
#     cmap="bwr",
#     alpha=0.5,
# )
# axs[1].scatter(
#     X_test[:, 0].cpu().numpy(),
#     X_test[:, 1].cpu().numpy(),
#     c=y_test.cpu().numpy(),
#     cmap="bwr",
#     alpha=0.5,
# )
# axs[1].set_title("Figure 2")

# # Display the figure in Streamlit
# # st.subheader("One Figure with Two Subplots:")
# st.pyplot(fig)


n_samples = 5

n_grid = 200
lims = 4
tot_itr = 6
twod_grid = (
    torch.tensor(
        np.meshgrid(np.linspace(-lims, lims, n_grid), np.linspace(-lims, lims, n_grid))
    )
    .float()
)
y_preds = []
acc = []
acc = torch.zeros(int((tot_itr)))

for i in range(n_samples):
    m = GPy.models.GPClassification(X_train.detach().numpy(), y_train.detach().numpy())
    out_pred = m.predict(X_test.cpu().numpy())
    pred = out_pred[0].flatten() > 0.5
    acc[0] = (torch.tensor(pred) == y_test.flatten()).sum().float() / y_test.shape[0]
    for itr in range(1, tot_itr):
        m.optimize(
            "bfgs", max_iters=10
        )  # first runs EP and then optimizes the kernel parameters
        print("iteration:", itr)
        print(m)
        print("")
        out_pred = m.predict(X_test.cpu().numpy())
        pred = out_pred[0].flatten() > 0.5
        acc[itr] = (
            torch.tensor(pred) == y_test.flatten()
        ).sum().float() / y_test.shape[0]

    simY, simMse = m.predict(
        twod_grid.view(2, -1).T.detach().numpy()
    )  # (twod_grid.view(2, -1).T)
    y_preds.append(simY)

plt.figure(figsize=(10, 5))
plt.plot(acc, label="Accuracy")
plt.grid()
plt.xlabel("Iteration number")
plt.ylabel("Sample accuracy")
plt.tick_params(labelsize=15)
plt.legend()
st.pyplot(plt)
