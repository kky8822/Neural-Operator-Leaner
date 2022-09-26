import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

ftotal_list = [
    "result_GT3d_ns_V1000_Ffix_T1024_V200_10to40_500.pt",
    "result_GT3d_ns_V10000_Ffix_T1024_V200_10to40_500.pt",
    "result_FNO3d_ns_V1000_Ffix_T1024_V200_10to40_500.pt",
    "result_FNO3d_ns_V10000_Ffix_T1024_V200_10to40_500.pt",
    "result_AFNO3d_ns_V1000_Ffix_T1024_V200_10to40_500.pt",
    "result_AFNO3d_ns_V10000_Ffix_T1024_V200_10to40_500.pt",
    # "result_UNO_ns_V1000_Ffix_T1024_V200_10to40_500_norm.pt",
    "result_UNO_ns_V1000_Ffix_T1024_V200_10to40_500.pt",
    # "result_UNO_ns_V10000_Ffix_T1024_V200_10to40_500_norm.pt",
    "result_UNO_ns_V10000_Ffix_T1024_V200_10to40_500.pt",
    "result_PDNO_ns_V1000_Ffix_T1024_V200_10to40_500.pt",
    "result_PDNO_ns_V10000_Ffix_T1024_V200_10to40_500.pt",
]
c_list = ["r", "g", "b", "m", "k"]


def log_fit(x, a, b):
    return -a * np.log(x) / np.log(0.01) + b


# -----------------loss V1000 --------------------------#
plt.rcParams["figure.figsize"] = (11.693 / 4, 8.267 / 2)
plt.rcParams.update({"font.size": 12})

fig, ax = plt.subplots()
f_list = [x for x in ftotal_list if "V1000_" in x]
for i, f in enumerate(f_list):
    ff = open(f, "rb")
    data = pickle.load(ff)
    if "UNO" in f:
        plt.plot(range(len(data["var_l2"]))[0::2], data["var_l2"][0::2], "-", color=c_list[i], label=f)
        plt.plot(range(len(data["train_l2"]))[0::2], data["train_l2"][0::2], "--", color=c_list[i], label=f)
    elif "PDNO" in f:
        x = [0, 0.33861262, 4.1678047, 8.751128, 18.622547, 29.238914, 26.952417, 66.37496, 113.36751, 167.92891, 230.81444, 289.15796, 324.00867, 366.44086, 405.84042, 445.99185, 499]
        y = [data["var_l2"][0].item(), 0.106611304, 0.067273445, 0.044112258, 0.034163054, 0.030456087, 0.035507172, 0.025814502, 0.02001122, 0.016753333, 0.015540546, 0.0138711985, 0.013532872, 0.012387172, 0.011631397, 0.011643351, data["var_l2"][-1].item()]
        f = interp1d(x, y)
        x_fit = np.linspace(0, 499, 500)
        plt.plot(x_fit, f(x_fit), "-", color=c_list[i], label=f)

        x = [0, 0.33861262, 4.1678047, 8.751128, 18.622547, 29.238914, 27.718027, 66.38529, 113.37899, 167.9381, 233.09749, 290.6869, 325.53644, 366.4466, 407.36362, 458.12796, 499]
        y = [data["train_l2"][0].item(), 0.106611304, 0.067273445, 0.044112258, 0.034163054, 0.030456087, 0.032466486, 0.023007177, 0.017608305, 0.015123642, 0.013851318, 0.011897716, 0.011756964, 0.01161968, 0.010635539, 0.0098625645, data["train_l2"][-1].item()]
        f = interp1d(x, y)
        x_fit = np.linspace(0, 499, 500)
        plt.plot(x_fit, f(x_fit), "--", color=c_list[i], label=f)

        # x = [1, 499]
        # y = data["var_l2"]
        # x_fit = np.linspace(1, 500, 499)
        # b = y[-1].item()
        # popt, pcov = curve_fit(lambda x, a: log_fit(x, a, b), x, y)
        # print(popt)
        # plt.plot(x_fit, log_fit(x_fit, *popt, b), "-", color=c_list[i], label=f)

        # x = [1, 499]
        # y = data["train_l2"]
        # x_fit = np.linspace(1, 500, 499)
        # b = y[-1].item()
        # popt, pcov = curve_fit(lambda x, a: log_fit(x, a, b), x, y)
        # plt.plot(x_fit, log_fit(x_fit, *popt, b), "--", color=c_list[i], label=f)
    else:
        try:
            plt.plot(range(len(data["loss_val"])), data["loss_val"], "-", color=c_list[i], label=f)
            plt.plot(range(len(data["loss_train"])), data["loss_train"][:, 0], "--", color=c_list[i], label=f)
        except:
            plt.plot(range(len(data["var_l2"])), data["var_l2"], "-", color=c_list[i], label=f)
            plt.plot(range(len(data["train_l2"])), data["train_l2"], "--", color=c_list[i], label=f)

# plt.yscale("log")
plt.ylim([0, 1])
plt.xlim([0, 500])
plt.xticks([0, 100, 200, 300, 400, 500])
plt.grid()
fig.tight_layout()

# plt.ylim([0, 1])
# plt.legend()
plt.savefig("V1000_loss_epoch.png")

# -----------------loss V10000 --------------------------#
plt.clf()
plt.cla()
f_list = [x for x in ftotal_list if "V10000_" in x]
for i, f in enumerate(f_list):
    ff = open(f, "rb")
    data = pickle.load(ff)
    if "UNO" in f:
        plt.plot(range(len(data["var_l2"]))[0::2], data["var_l2"][0::2], "-", color=c_list[i], label=f)
        plt.plot(range(len(data["train_l2"]))[0::2], data["train_l2"][0::2], "--", color=c_list[i], label=f)
    elif "PDNO" in f:
        # plt.plot([0, 499], data["var_l2"], "-", color=c_list[i], label=f)
        # plt.plot([0, 499], data["train_l2"], "--", color=c_list[i], label=f)

        x = [0, 1.1742607, 4.2190485, 7.8157096, 11.549957, 18.18827, 26.900698, 42.042076, 52.20581, 64.6505, 130.19229, 232.23894, 347.9743, 463.7094, 496.47998, 499]
        y = [data["var_l2"][0].item(), 0.5817236, 0.5348325, 0.5104443, 0.49751627, 0.48042014, 0.46392047, 0.45913568, 0.45120695, 0.4512594, 0.45471027, 0.45196638, 0.45563623, 0.4625652, 0.46923578, data["var_l2"][-1].item()]
        f = interp1d(x, y)
        x_fit = np.linspace(0, 499, 500)
        plt.plot(x_fit, f(x_fit), "-", color=c_list[i], label=f)

        x = [0, 0.30870658, 2.3911417, 8.620713, 26.679102, 44.734566, 73.57703, 100.75853, 145.36775, 201.5947, 300.14578, 381.26865, 454.09225, 497.03912, 499]
        y = [data["train_l2"][0].item(), 0.593237, 0.50669587, 0.45297974, 0.38026366, 0.33881778, 0.31052017, 0.29164726, 0.27207458, 0.2565399, 0.23950434, 0.22749723, 0.21835068, 0.21468695, data["train_l2"][-1].item()]
        f = interp1d(x, y)
        x_fit = np.linspace(0, 499, 500)
        plt.plot(x_fit, f(x_fit), "--", color=c_list[i], label=f)

        # x = [1, 499]
        # y = data["var_l2"]
        # x_fit = np.linspace(1, 500, 499)
        # b = y[-1].item()
        # popt, pcov = curve_fit(lambda x, a: log_fit(x, a, b), x, y)
        # plt.plot(x_fit, log_fit(x_fit, *popt, b), "-", color=c_list[i], label=f)

        # x = [1, 499]
        # y = data["train_l2"]
        # x_fit = np.linspace(1, 500, 499)
        # b = y[-1].item()
        # popt, pcov = curve_fit(lambda x, a: log_fit(x, a, b), x, y)
        # plt.plot(x_fit, log_fit(x_fit, *popt, b), "--", color=c_list[i], label=f)
    else:
        try:
            plt.plot(range(len(data["loss_val"])), data["loss_val"], "-", color=c_list[i], label=f)
            plt.plot(range(len(data["loss_train"])), data["loss_train"][:, 0], "--", color=c_list[i], label=f)
        except:
            plt.plot(range(len(data["var_l2"])), data["var_l2"], "-", color=c_list[i], label=f)
            plt.plot(range(len(data["train_l2"])), data["train_l2"], "--", color=c_list[i], label=f)

# plt.yscale("log")
plt.ylim([0, 1])
# plt.yticks([1e-1, 1])
# plt.ylim([0, 1])
# plt.legend()
plt.xlim([0, 500])
plt.xticks([0, 100, 200, 300, 400, 500])
plt.grid()
plt.savefig("V10000_loss_epoch.png")

# -----------------time_loss V1000 --------------------------#
plt.clf()
plt.cla()
f_list = [x for x in ftotal_list if "V1000_" in x]
for i, f in enumerate(f_list):
    ff = open(f, "rb")
    data = pickle.load(ff)

    if "GT" in f:
        test_l2_time = [0.00788668, 0.00636564, 0.00569404, 0.00544315, 0.00545911, 0.00529859, 0.00531527, 0.00558758, 0.00572988, 0.00614722, 0.00657373, 0.00705855, 0.00776053, 0.00863351, 0.00950179, 0.01029788, 0.011198, 0.01226973, 0.01320457, 0.01429891, 0.01567752, 0.01697929, 0.01797716, 0.01902462, 0.02027846, 0.0217116, 0.02294055, 0.02375413, 0.02495644, 0.02683923, 0.02859974, 0.02956801, 0.03072047, 0.03266639, 0.03457784, 0.03593041, 0.0372024, 0.03973199, 0.04399373, 0.05236697]
        plt.plot(range(len(test_l2_time)), test_l2_time, "-", color=c_list[i], label=f)
    else:
        plt.plot(range(len(data["test_l2_time"])), data["test_l2_time"], "-", color=c_list[i], label=f)

# plt.yscale("log")
# plt.ylim([1e-3, 1])
plt.ylim([0, 1])
# plt.legend()
plt.xlim([0, 40])
plt.xticks([0, 10, 20, 30, 40])
plt.grid()
plt.savefig("V1000_time_loss.png")

# -----------------time_loss V10000 --------------------------#
plt.clf()
plt.cla()
f_list = [x for x in ftotal_list if "V10000_" in x]
for i, f in enumerate(f_list):
    ff = open(f, "rb")
    data = pickle.load(ff)

    if "GT" in f:
        test_l2_time = [0.03549062, 0.03391174, 0.0359607, 0.04072488, 0.04790405, 0.0566822, 0.06818202, 0.08219776, 0.09907705, 0.11800604, 0.14019549, 0.16604671, 0.19028115, 0.21258579, 0.22802494, 0.24240768, 0.26267087, 0.28380504, 0.3184037, 0.36494467, 0.42748162, 0.488506, 0.53079945, 0.55681795, 0.5779258, 0.60574514, 0.6412691, 0.67636055, 0.70870394, 0.729554, 0.7428372, 0.7539558, 0.770332, 0.78830284, 0.80677384, 0.8219925, 0.834775, 0.8416078, 0.84575003, 0.8523076]
        plt.plot(range(len(test_l2_time)), test_l2_time, "-", color=c_list[i], label=f)
    else:
        plt.plot(range(len(data["test_l2_time"])), data["test_l2_time"], "-", color=c_list[i], label=f)

# plt.yscale("log")
# plt.ylim([1e-3, 1])
plt.ylim([0, 1])
# plt.legend()
plt.xlim([0, 40])
plt.xticks([0, 10, 20, 30, 40])
plt.grid()
plt.savefig("V10000_time_loss.png")
