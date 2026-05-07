import json
import numpy as np
import matplotlib.pyplot as plt

TRIALNAME = "uwb_ssr_test1"
IN_PATH = f"/home/antond2/Desktop/Research/MultiXR-Post/2/collect/{TRIALNAME}_nuc2_raw/uwb_raw.json"

#IN_PATH = "uwb_raw.json"

GT = 0.5 #m
UUS_TO_DWT_TIME = 65536
SPEED_OF_LIGHT = 299702547 # m/s
DWT_TIME_UNITS = 1.0 / 499.2e6 / 128.0 #!< = 15.65e-12 s

# All timestamps are in dtu
def to_s(dtu_ts): return dtu_ts * DWT_TIME_UNITS
    
all = json.load(open(IN_PATH, 'r'))

# Validating + cleaning reported timestamps

ranges = [] # Just the json objects
reported_ranges = []
calculated_ranges = []
avg_hold_time_src = []
avg_hold_time_dst = []
overflow_errors = 0

for i, r in enumerate(all):
    if r["type"] == "uwb":
        ascending_src = r["poll_src_tx"] < r["resp_src_rx"] < r["final_src_tx"]
        ascending_dst = r["poll_dst_rx"] < r["resp_dst_tx"] < r["final_dst_rx"]

        # To ensure that all equations are in a consistent frame
        # i.e. some equations don't suggest opposite skew drift, this happens whenever timestamps overflow at different times
        # between src and dst
        dst_greater_than_src = to_s(r["poll_src_tx"]) + 5e-3 < to_s(r["poll_dst_rx"])

        if ascending_src and ascending_dst and dst_greater_than_src:
            ranges.append(r)

            poll_src_tx = to_s(r["poll_src_tx"])
            resp_src_rx   = to_s(r["resp_src_rx"])
            final_src_tx  = to_s(r["final_src_tx"])
            poll_dst_rx   = to_s(r["poll_dst_rx"])
            resp_dst_tx   = to_s(r["resp_dst_tx"])
            final_dst_rx  = to_s(r["final_dst_rx"])
    
            avg_hold_time_src.append(final_src_tx - resp_src_rx)
            avg_hold_time_dst.append(resp_dst_tx - poll_dst_rx)

            # Note timestamps already in s so tof is in s
            roundB = (final_dst_rx - resp_dst_tx)
            replyB = (resp_dst_tx - poll_dst_rx)
            roundA = (resp_src_rx - poll_src_tx)
            replyA = (final_src_tx - resp_src_rx)
            tof_s = ((roundA * roundB - replyA * replyB) /
                        (roundA + roundB + replyA + replyB))
            calculated_ranges.append(tof_s * SPEED_OF_LIGHT )
            reported_ranges.append(r["range"])
        else:
            overflow_errors +=1


print(f"Validating ROS reported timestamps")
print(f"{overflow_errors} overflows / {len(all)} ranges")

reported_ranges, calculated_ranges = (np.array(reported_ranges), np.array(calculated_ranges))

print(f" Range reported vs range computed with timestamps should have same error ")
avg_reported_range_err = np.mean(np.abs(reported_ranges - GT))
print(f"{avg_reported_range_err=}")
avg_calculated_range_err = np.mean(np.abs(calculated_ranges - GT))
print(f"{avg_calculated_range_err=}")

print(f" Message hold time at initiator, should be ~2.0ms = { np.mean(np.array(avg_hold_time_src))=}")
print(f" Message hold time at responder, should be ~1.5ms = { np.mean(np.array(avg_hold_time_dst))=}")


import matplotlib.pyplot as plt

def plot_distribution(data, title, xlabel, ylabel):

    mean = np.mean(data)
    std = np.std(data)

    plt.hist(data, bins='auto')

    # Mean line
    plt.axvline(mean, linestyle='--')

    plt.title(
        f"{title}\n"
        f"mean={mean:.3e}, std={std:.3e}"
    )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.margins(x=0.05)
    plt.grid(True)

    plt.show()


    ### Compute SSR results:

def evaluate_ssr(D_dst_src, S_dst_src, tof_ = GT / SPEED_OF_LIGHT):

    ssr_ranges = []

    tofs = []
    poll_src_tx_ = []
    poll_dst_rx_ = []
    base_ranges = []

    
    for i, r in enumerate(ranges):
    
        # Now we treat each double sided two way range as 
        # a single sided range by just looking at the poll message
        poll_src_tx = to_s(r["poll_src_tx"])
        poll_dst_rx   = to_s(r["poll_dst_rx"])
    
        poll_src_rx = (1 + D_dst_src)*poll_dst_rx + S_dst_src
    
        tof_est = poll_src_rx - poll_src_tx
        tofs.append(tof_est)
        
        base_ranges.append(SPEED_OF_LIGHT * (poll_dst_rx - poll_src_tx))
        poll_src_tx_.append(poll_src_tx)
        poll_dst_rx_.append(poll_dst_rx)
        
        range_est = SPEED_OF_LIGHT * tof_est
        
        ssr_ranges.append(range_est)
        
    # plt.plot(tofs)
    # plt.title("SSR ToF Estimates (s)")
    # plt.show()
    
    return eval_error(ssr_ranges, reported_ranges, base_ranges)

def eval_error(ssr_ranges, reported_ranges, base_ranges=None):

    reported_ranges = np.array(reported_ranges)
    ssr_ranges = np.array(ssr_ranges)

    dstwr_err = np.abs(reported_ranges - GT)
    avg_dstwr_err = np.mean(dstwr_err)
    std_dstwr_err = np.std(dstwr_err)

    ssr_err = np.abs(ssr_ranges - GT)

    # plt.plot(ssr_err)
    # plt.title("Single Side Ranging Error (m)")
    # plt.show()

    if base_ranges is not None:
        base_ranges = np.array(base_ranges)
        base_err = np.abs(base_ranges - GT)
        # plt.plot(base_err)
        # plt.title("Base Single Side Ranging Error (m)")
        # plt.show()

        # fig, ax = plt.subplots()
        # ax.plot(ssr_err, label='SSR')
        # ax.plot(base_err, label='Base')
        # ax.legend()
        # plt.show()
    
    avg_ssr_err = np.mean(ssr_err)
    std_ssr_err = np.std(ssr_err)

    # print(f"{avg_dstwr_err=}\n {std_dstwr_err=}")
    # print(f"{avg_ssr_err=}\n {std_ssr_err=}")
    
    return avg_dstwr_err, avg_ssr_err


### Using 3 equations and least squares

def three_eq_ls_est_tof(tof_=GT / SPEED_OF_LIGHT):

    drift_dst_src = [] # drift from destination to source
    skew_dst_src = []
    tofs = []
    
    for r in ranges:
        
        poll_src_tx = to_s(r["poll_src_tx"])
        resp_src_rx   = to_s(r["resp_src_rx"])
        final_src_tx  = to_s(r["final_src_tx"])
        poll_dst_rx   = to_s(r["poll_dst_rx"])
        resp_dst_tx   = to_s(r["resp_dst_tx"])
        final_dst_rx  = to_s(r["final_dst_rx"])
        
        A = np.array(
                    [[ poll_dst_rx, 1, -1],
                      [resp_dst_tx, 1, 1],
                      [final_dst_rx, 1, -1]]
                    )
        b = np.array([ poll_src_tx - poll_dst_rx,
                     resp_src_rx - resp_dst_tx,
                     final_src_tx - final_dst_rx ])
        
        x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
        drift_dst_src.append(x[0])
        skew_dst_src.append(x[1])
        tofs.append(x[2])

    return np.array(drift_dst_src), np.array(skew_dst_src), np.array(tofs)

print(f"Analyzing estimated D, S")

drift_dst_src, skew_dst_src, tofs = three_eq_ls_est_tof()

# Apply a second linear fit to re-estimate drift

d_s_corr = np.corrcoef(drift_dst_src, skew_dst_src)[0,1]
print(f"{d_s_corr=}")

d_tof_corr = np.corrcoef(drift_dst_src, tofs)[0,1]
print(f"{d_tof_corr=}")


s_tof_corr = np.corrcoef(skew_dst_src, tofs)[0,1]
print(f"{s_tof_corr=}")


print(f"Using D, S for SSR")

D_dst_src_sol = np.mean(drift_dst_src)
S_dst_src_sol = np.mean(skew_dst_src)

D_dst_src = 0
S_dst_src = 0


# Center search around current estimate
D_vals = np.linspace(D_dst_src - 1e-1,
                     D_dst_src + 1e-1,
                     1000)

S_vals = np.linspace(S_dst_src - 1e-1,
                     S_dst_src + 1e-1,
                     1000)

loss = np.zeros((len(S_vals), len(D_vals)))

for i, S in enumerate(S_vals):
    for j, D in enumerate(D_vals):

        _, ssr_err = evaluate_ssr(D, S)

        loss[i, j] = ssr_err


D_grid, S_grid = np.meshgrid(D_vals, S_vals)

plt.figure(figsize=(8, 6))

contours = plt.contourf(
    D_grid,
    S_grid,
    loss,
    levels=50
)

plt.colorbar(contours, label="SSR Error")

plt.xlabel("Drift D")
plt.ylabel("Skew S")

plt.title("SSR Loss Landscape")

# mark current estimate
plt.plot(D_dst_src_sol, S_dst_src_sol, 'rx', markersize=12)

plt.show()


