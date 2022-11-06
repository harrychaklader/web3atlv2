import datetime
import numpy as np
from pathlib import Path
import os
import time

import matplotlib
import matplotlib.pyplot as plt

from ocean_lib.example_config import ExampleConfig
from ocean_lib.ocean.ocean import Ocean
from ocean_lib.web3_internal.wallet import Wallet


def create_ocean_instance() -> Ocean:
  config = ExampleConfig.get_config(
    "https://polygon-rpc.com")  # points to Polygon mainnet
  config["BLOCK_CONFIRMATIONS"] = 1  #faster
  ocean = Ocean(config)
  return ocean


def create_alice_wallet(ocean: Ocean) -> Wallet:
  config = ocean.config_dict
  alice_private_key = "8dac11d5d6e185a9dcff61d090a8e1fba087bd1eedf18876aa7fd663b0b5e64a"

  alice_wallet = Wallet(ocean.web3, alice_private_key,
                        config["BLOCK_CONFIRMATIONS"],
                        config["TRANSACTION_TIMEOUT"])
  bal = ocean.from_wei(alice_wallet.web3.eth.get_balance(alice_wallet.address))
  print(f"alice_wallet.address={alice_wallet.address}. bal={bal}")
  print(bal)
  assert bal > 0, f"Alice needs MATIC"
  return alice_wallet


#helper functions: time
def to_unixtime(dt: datetime.datetime):
  return time.mktime(dt.timetuple())


def to_unixtimes(dts: list) -> list:
  return [to_unixtime(dt) for dt in dts]


def to_datetime(ut) -> datetime.datetime:
  return datetime.datetime.utcfromtimestamp(ut)


def to_datetimes(uts: list) -> list:
  return [to_datetime(ut) for ut in uts]


def round_to_nearest_hour(dt: datetime.datetime) -> datetime.datetime:
  return (dt.replace(second=0, microsecond=0, minute=0, hour=dt.hour) +
          datetime.timedelta(hours=dt.minute // 30))


def pretty_time(dt: datetime.datetime) -> str:
  return dt.strftime('%Y/%m/%d, %H:%M:%S')


def print_datetime_info(descr: str, uts: list):
  dts = to_datetimes(uts)
  print(descr + ":")
  print(f"  starts on: {pretty_time(dts[0])}")
  print(f"    ends on: {pretty_time(dts[-1])}")
  print(f"  {len(dts)} datapoints")
  print(f"  time interval between datapoints: {(dts[1]-dts[0])}")


def target_12h_unixtimes(start_dt: datetime.datetime) -> list:
  target_dts = [start_dt + datetime.timedelta(hours=h) for h in range(12)]
  target_uts = to_unixtimes(target_dts)
  return target_uts


#helper-functions: higher level
def load_from_ohlc_data(file_name: str) -> tuple:
  """Returns (list_of_unixtimes, list_of_close_prices)"""
  with open(file_name, "r") as file:
    data_str = file.read().rstrip().replace('"', '')
  x = eval(data_str)  #list of lists
  uts = [xi[0] / 1000 for xi in x]
  vals = [xi[4] for xi in x]
  return (uts, vals)


def filter_to_target_uts(target_uts: list, unfiltered_uts: list,
                         unfiltered_vals: list) -> list:
  """Return filtered_vals -- values at at the target timestamps"""
  filtered_vals = [None] * len(target_uts)
  for i, target_ut in enumerate(target_uts):
    time_diffs = np.abs(np.asarray(unfiltered_uts) - target_ut)
    tol_s = 1  #should always align within e.g. 1 second
    target_ut_s = pretty_time(to_datetime(target_ut))
    assert min(time_diffs) <= tol_s, \
        f"Unfiltered times is missing target time: {target_ut_s}"
    j = np.argmin(time_diffs)
    filtered_vals[i] = unfiltered_vals[j]
  return filtered_vals


#helpers: save/load list
def save_list(list_: list, file_name: str):
  """Save a file shaped: [1.2, 3.4, 5.6, ..]"""
  p = Path(file_name)
  p.write_text(str(list_))


def load_list(file_name: str) -> list:
  """Load from a file shaped: [1.2, 3.4, 5.6, ..]"""
  p = Path(file_name)
  s = p.read_text()
  list_ = eval(s)
  return list_


#helpers: prediction performance
def calc_nmse(y, yhat) -> float:
  assert len(y) == len(yhat)
  mse_xy = np.sum(np.square(np.asarray(y) - np.asarray(yhat)))
  mse_x = np.sum(np.square(np.asarray(y)))
  nmse = mse_xy / mse_x
  return nmse


def plot_prices(cex_vals, pred_vals):
  matplotlib.rcParams.update({'font.size': 22})

  x = [h for h in range(0, 12)]
  assert len(x) == len(cex_vals) == len(pred_vals)

  fig, ax = plt.subplots()
  ax.plot(x, cex_vals, '--', label="CEX values")
  ax.plot(x, pred_vals, '-', label="Pred. values")
  ax.legend(loc='lower right')
  plt.ylabel("ETH price")
  plt.xlabel("Hour")
  fig.set_size_inches(18, 18)
  plt.xticks(x)
  plt.show()


# Data set
pred_vals = [
1619.3577,
1619.9607,
1620.6091,
1621.419,
1622.3264,
1623.4308,
1624.6201,
1625.8335,
1627.1067,
1628.3867,
1629.7463,
1631.1682,
]

#File and save list as csv
file_name = "/tmp/pred_vals.csv"
save_list(pred_vals, file_name)


ocean = create_ocean_instance()
alice_wallet = create_alice_wallet(ocean) #you're Alice

#File and save list as csv
file_name = "/tmp/pred_vals.csv"
save_list(pred_vals, file_name)

from pybundlr import pybundlr
file_name =  "/tmp/pred_vals.csv"
url = pybundlr.fund_and_upload(file_name, "matic", "8dac11d5d6e185a9dcff61d090a8e1fba087bd1eedf18876aa7fd663b0b5e64a")
url = "https://arweave.net/tx/brV48kL4v3nUl6RNTFxZkkYg1m-u9AaXaS98mOgs0Mw"
print(f"Your csv url: {url}")

name = "ETH predictions " + str(time.time()) #time for unique name
(data_nft, datatoken, asset) = ocean.assets.create_url_asset(name, url, alice_wallet, wait_for_aqua=False)
data_nft.set_metadata_state(metadata_state=5, from_wallet=alice_wallet)
print(f"New asset created, with did={asset.did}, and datatoken.address={datatoken.address}")
