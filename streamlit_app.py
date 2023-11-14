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

st.header("Bayesian Deep Learning for Classification")


def generate_make_moons(seed):
    X, y = make_moons(n_samples=1000, noise=0.3, random_state=seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )
    return X_train, X_test, y_train, y_test


st.title("Genaerate Make Moons Dataset with different seeds")

seed = st.slider("Select Seed:", min_value=0, max_value=100, value=43)

X_train, X_test, y_train, y_test = generate_make_moons(seed)


# st.subheader("Visualization:")
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


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        #         x = F.softmax(self.fc3(x))
        return x


net = Net()

tau_list = []
tau = 10.0  # ./100. # 1/50
for w in net.parameters():
    tau_list.append(tau)
tau_list = torch.tensor(tau_list)

hamiltorch.set_random_seed(seed)

params_init = hamiltorch.util.flatten(net)
print(params_init.shape)

step_size = 0.001
num_samples = 300
L = 20
tau_out = 1.0
normalizing_const = 1.0
burn = 100

params_hmc = hamiltorch.sample_model(
    net,
    X_train,
    y_train,
    params_init=params_init,
    model_loss="multi_class_linear_output",
    num_samples=num_samples,
    burn=burn,
    step_size=step_size,
    num_steps_per_sample=L,
    tau_out=tau_out,
    tau_list=tau_list,
    normalizing_const=normalizing_const,
)


pred_list, log_prob_list = hamiltorch.predict_model(
    net,
    x=X_test,
    y=y_test,
    samples=params_hmc,
    model_loss="multi_class_log_softmax_output",
    tau_out=1.0,
    tau_list=tau_list,
)
_, pred = torch.max(pred_list, 2)
acc = []
acc = torch.zeros(int(len(params_hmc)) - 1)
nll = torch.zeros(int(len(params_hmc)) - 1)
ensemble_proba = F.softmax(pred_list[0], dim=-1)
mcce = MulticlassCalibrationError(num_classes=2, n_bins=2)  # , norm='l2')

for s in range(1, len(params_hmc)):
    _, pred = torch.max(pred_list[:s].mean(0), -1)
    acc[s - 1] = (pred.float() == y_test.flatten()).sum().float() / y_test.shape[0]
    ensemble_proba += F.softmax(pred_list[s], dim=-1)
    #     print(ensemble_proba.cpu()/(s+1),y_test[:].long().cpu().flatten())

    out_calibration_error = mcce(
        ensemble_proba.cpu() / (s + 1), y_test[:].long().cpu().flatten()
    )
    print(out_calibration_error)
    nll[s - 1] = F.nll_loss(
        torch.log(ensemble_proba.cpu() / (s + 1)),
        y_test[:].long().cpu().flatten(),
        reduction="mean",
    )

st.subheader("Bayesian Neural Network using Hamiltorch")
# Create separate plots for accuracy and negative log likelihood
fig, (ax_acc, ax_nll) = plt.subplots(1, 2, figsize=(12, 5))

# Plot Accuracy
ax_acc.plot(acc)
ax_acc.set_title("Accuracy")
ax_acc.set_xlabel("Number of Samples")
ax_acc.set_ylabel("Accuracy")

# Plot Negative Log Likelihood
ax_nll.plot(nll)
ax_nll.set_title("Negative Log Likelihood")
ax_nll.set_xlabel("Number of Samples")
ax_nll.set_ylabel("NLL")

fig.tight_layout()
st.subheader("Model Evaluation: Baysian Neural Network")
st.pyplot(fig)


posterior_samples = params_hmc  # .detach()
# Consider burning the first 100 samples
posterior_samples = posterior_samples  # [1000:]
y_preds = []
n_grid = 200
lims = 4
twod_grid = torch.tensor(
    np.meshgrid(np.linspace(-lims, lims, n_grid), np.linspace(-lims, lims, n_grid))
).float()
with torch.no_grad():
    for theta in posterior_samples:
        params_list = hamiltorch.util.unflatten(net, theta)
        params = net.state_dict()
        for i, (name, _) in enumerate(params.items()):
            params[name] = params_list[i]
        y_pred = torch.func.functional_call(
            net, params, twod_grid.view(2, -1).T
        ).squeeze()

        y_preds.append(y_pred[:, 0])

logits = torch.stack(y_preds).mean(axis=0).reshape(n_grid, n_grid)
probs = torch.sigmoid(logits)


fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Plot the first subplot
axs[0].contourf(
    twod_grid[0].cpu().numpy(),
    twod_grid[1].cpu().numpy(),
    probs.cpu().numpy(),
    cmap="bwr",
    alpha=0.5,
)
axs[0].scatter(
    X_test[:, 0].cpu().numpy(),
    X_test[:, 1].cpu().numpy(),
    c=y_test.cpu().numpy(),
    cmap="bwr",
    alpha=0.5,
)
axs[0].set_title("Bayesian Neural Network: Mean value prediction")

# Plot the second subplot
axs[1].contourf(
    twod_grid[0].cpu().numpy(),
    twod_grid[1].cpu().numpy(),
    torch.stack(y_preds).std(axis=0).reshape(n_grid, n_grid).cpu().numpy(),
    cmap="bwr",
    alpha=0.5,
)
axs[1].scatter(
    X_test[:, 0].cpu().numpy(),
    X_test[:, 1].cpu().numpy(),
    c=y_test.cpu().numpy(),
    cmap="bwr",
    alpha=0.5,
)
axs[1].set_title("Bayesian Neural Network: Variance/Uncertainty in prediction")

# Display the figure in Streamlit
# st.subheader("One Figure with Two Subplots:")
st.pyplot(fig)


st.subheader("Bayesian Logistic Regression using Hamiltorch")


class Net(nn.Module):
    def __init__(self, layer_sizes, loss="multi_class", bias=True):
        super(Net, self).__init__()
        self.layer_sizes = layer_sizes
        self.layer_list = []
        self.loss = loss
        self.bias = bias
        self.l1 = nn.Linear(layer_sizes[0], layer_sizes[1], bias=self.bias)

    def forward(self, x):
        x = self.l1(x)

        return x


layer_sizes = [2, 2]
net = Net(layer_sizes)


print(net)
## Set hyperparameters for network

tau_list = []
tau = 1.0  # /100. # iris 1/10
for w in net.parameters():
    tau_list.append(tau)
tau_list = torch.tensor(tau_list)

hamiltorch.set_random_seed(seed)

params_init = hamiltorch.util.flatten(net)

step_size = 0.1
num_samples = 300
L = 20
tau_out = 1.0

params_hmc = hamiltorch.sample_model(
    net,
    X_train,
    y_train,
    params_init=params_init,
    num_samples=num_samples,
    step_size=step_size,
    num_steps_per_sample=L,
    tau_out=tau_out,
    tau_list=tau_list,
)

pred_list, log_prob_list = hamiltorch.predict_model(
    net,
    x=X_test,
    y=y_test,
    samples=params_hmc[:],
    model_loss="multi_class_linear_output",
    tau_out=1.0,
    tau_list=tau_list,
)
_, pred = torch.max(pred_list, 2)
acc = torch.zeros(len(pred_list) - 1)
nll = torch.zeros(len(pred_list) - 1)
ensemble_proba = F.softmax(pred_list[0], dim=-1)

mcce = MulticlassCalibrationError(num_classes=2, n_bins=2)  # , norm='l2')

for s in range(1, len(pred_list)):
    _, pred = torch.max(pred_list[:s].mean(0), -1)
    acc[s - 1] = (pred.float() == y_test.flatten()).sum().float() / y_test.shape[0]

    ensemble_proba += F.softmax(pred_list[s], dim=-1)
    #     print(ensemble_proba.cpu()/(s+1),y_test[:].long().cpu().flatten())

    out_calibration_error = mcce(
        ensemble_proba.cpu() / (s + 1), y_test[:].long().cpu().flatten()
    )
    print(out_calibration_error)

    nll[s - 1] = F.nll_loss(
        torch.log(ensemble_proba.cpu() / (s + 1)),
        y_test[:].long().cpu().flatten(),
        reduction="mean",
    )

fig, axs = plt.subplots(1, 2, figsize=(15, 6))

# Plot the first subplot (Accuracy)

axs[0].plot(acc, label="Accuracy")
axs[0].grid()
axs[0].set_xlabel("Iteration number")
axs[0].set_ylabel("Sample accuracy")
axs[0].tick_params(labelsize=15)
axs[0].legend()

# Plot the second subplot (Negative Log Likelihood)

axs[1].plot(nll, label="Loss")
axs[1].grid()
axs[1].set_xlabel("Iteration number")
axs[1].set_ylabel("Negative Log Likelihood")
axs[1].tick_params(labelsize=15)
axs[1].legend()

# Adjust layout
fig.tight_layout()

# Display the figure in Streamlit
st.subheader("Model Evaluation: Bayesian Logistic Regression")
st.pyplot(fig)

# Get posterior predictive over the 2D grid
posterior_samples = params_hmc  # .detach()
# Consider burning the first 100 samples
posterior_samples = posterior_samples  # [1000:]
y_preds = []
n_grid = 200
lims = 4
twod_grid = torch.tensor(
    np.meshgrid(np.linspace(-lims, lims, n_grid), np.linspace(-lims, lims, n_grid))
).float()
with torch.no_grad():
    for theta in posterior_samples:
        params_list = hamiltorch.util.unflatten(net, theta)
        params = net.state_dict()
        for i, (name, _) in enumerate(params.items()):
            params[name] = params_list[i]
        y_pred = torch.func.functional_call(
            net, params, twod_grid.view(2, -1).T
        ).squeeze()

        y_preds.append(y_pred[:, 0])

logits = torch.stack(y_preds).mean(axis=0).reshape(n_grid, n_grid)
probs = torch.sigmoid(logits)

# Create one figure with two subplots
fig, axs = plt.subplots(1, 2, figsize=(15, 6))

# Plot the first subplot
axs[0].contourf(
    twod_grid[0].cpu().numpy(),
    twod_grid[1].cpu().numpy(),
    probs.cpu().numpy(),
    cmap="bwr",
    alpha=0.5,
)
scatter1 = axs[0].scatter(
    X_test[:, 0].cpu().numpy(),
    X_test[:, 1].cpu().numpy(),
    c=y_test.cpu().numpy(),
    cmap="bwr",
    alpha=0.5,
)
axs[0].set_xlabel("Feature 1")
axs[0].set_ylabel("Feature 2")
axs[0].set_title("Bayesian Logistic Regression: Mean value prediction")
axs[0].legend(handles=scatter1.legend_elements()[0], labels=["Class 1", "Class 0"])

# Plot the second subplot
axs[1].contourf(
    twod_grid[0].cpu().numpy(),
    twod_grid[1].cpu().numpy(),
    torch.stack(y_preds).std(axis=0).reshape(n_grid, n_grid).cpu().numpy(),
    cmap="bwr",
    alpha=0.5,
)
scatter2 = axs[1].scatter(
    X_test[:, 0].cpu().numpy(),
    X_test[:, 1].cpu().numpy(),
    c=y_test.cpu().numpy(),
    cmap="bwr",
    alpha=0.5,
)
axs[1].set_xlabel("Feature 1")
axs[1].set_ylabel("Feature 2")
axs[1].set_title("Bayesian Logistic Regression: Variance/Uncertainty in prediction")
axs[1].legend(handles=scatter2.legend_elements()[0], labels=["Class 1", "Class 0"])

# Adjust layout
fig.tight_layout()

# Display the figure in Streamlit
st.subheader("Mean and Variance/Uncertainty Plot: Bayesian Logistic Regression")
st.pyplot(fig)


st.subheader("Gaussian Processes using GPy")

n_samples = 3

n_grid = 200
lims = 4
tot_itr = 6
twod_grid = torch.tensor(
    np.meshgrid(np.linspace(-lims, lims, n_grid), np.linspace(-lims, lims, n_grid))
).float()
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


st.subheader("Model Evaluation: Gaussian Processes")
plt.figure(figsize=(4, 4))
plt.plot(acc, label="Accuracy")
plt.grid()
plt.xlabel("Iteration number")
plt.ylabel("Sample accuracy")
plt.tick_params(labelsize=15)
plt.legend()
st.pyplot(plt)

probs = 1 - np.stack(y_preds).mean(axis=0).reshape(n_grid, n_grid)

st.subheader("Mean and Variance/Uncertainty Plot: Gaussian Processes")

fig, axs = plt.subplots(1, 2, figsize=(10, 6))
axs[0].contourf(
    twod_grid[0].cpu().numpy(),
    twod_grid[1].cpu().numpy(),
    probs,
    cmap="bwr",
    alpha=0.5,
)
scatter1 = axs[0].scatter(
    X_test[:, 0].cpu().numpy(),
    X_test[:, 1].cpu().numpy(),
    c=y_test.cpu().numpy(),
    cmap="bwr",
    alpha=0.5,
)
axs[0].set_xlabel("Feature 1")
axs[0].set_ylabel("Feature 2")
axs[0].set_title("Gaussian Processes: Mean value prediction")
axs[0].legend(handles=scatter1.legend_elements()[0], labels=["Class 1", "Class 0"])

# Plot the second subplot
axs[1].contourf(
    twod_grid[0].cpu().numpy(),
    twod_grid[1].cpu().numpy(),
    np.stack(y_preds).std(axis=0).reshape(n_grid, n_grid),
    cmap="bwr",
    alpha=0.5,
)
scatter2 = axs[1].scatter(
    X_test[:, 0].cpu().numpy(),
    X_test[:, 1].cpu().numpy(),
    c=y_test.cpu().numpy(),
    cmap="bwr",
    alpha=0.5,
)
axs[1].set_xlabel("Feature 1")
axs[1].set_ylabel("Feature 2")
axs[1].set_title("Gaussian Processes: Variance/Uncertainty value prediction")
axs[1].legend(handles=scatter2.legend_elements()[0], labels=["Class 1", "Class 0"])

# Display the figure in Streamlit
st.pyplot(fig)
