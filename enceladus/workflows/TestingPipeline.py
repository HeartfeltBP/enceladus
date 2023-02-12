import wandb
import keras
import numpy as np
import pandas as pd
import pickle as pkl
from tqdm import tqdm
from typing import Tuple
from heartpy.preprocessing import flip_signal
from heartpy.peakdetection import detect_peaks
from heartpy.datautils import rolling_mean
from database_tools.tools.records import read_records, rescale_data


class TestingPipeline():
    def __init__(
        self,
        data_dir: str,
        model_dir: str,
        scaler: str,
    ):
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.model = self._load_model(model_dir)
        with open(data_dir + scaler, 'rb') as f:
            self.scaler = pkl.load(f)

    def _load_model(self, path):
        run = wandb.init()
        artifact = run.use_artifact(path, type='model')
        artifact_dir = artifact.download()
        wandb.finish()
        model = keras.models.load_model(artifact_dir)
        return model

    def run(self):
        print('Loading data...')
        data = read_records(data_dir=self.data_dir)
        ppg, vpg, apg, abp = self._load_data(data['test'])
        # return (ppg, vpg, apg, abp)

        print('Generating predictions...')
        pred = self.model.predict([ppg, vpg, apg]).reshape(-1, 256)
        # return pred

        print('Rescaling data...')
        ppg_scaled = rescale_data(ppg, self.scaler['ppg'])
        vpg_scaled = rescale_data(vpg, self.scaler['vpg'])
        apg_scaled = rescale_data(apg, self.scaler['apg'])
        abp_scaled = rescale_data(abp, self.scaler['abp'])
        pred_scaled = rescale_data(pred, self.scaler['abp'])
        # return (ppg_scaled, vpg_scaled, apg_scaled, abp_scaled, pred_scaled)

        print('Calculating error...')
        df = self._calculate_error(abp_scaled, pred_scaled)
        return df

    def _load_data(self, data):
        ppg, vpg, apg, abp = [], [], [], []
        for inputs, label in tqdm(data.as_numpy_iterator()):
            ppg.append(inputs['ppg'])
            vpg.append(inputs['vpg'])
            apg.append(inputs['apg'])
            abp.append(label)
        ppg = np.array(ppg).reshape(-1, 256)
        vpg = np.array(vpg).reshape(-1, 256)
        apg = np.array(apg).reshape(-1, 256)
        abp = np.array(abp).reshape(-1, 256)
        return ppg, vpg, apg, abp

    def _calculate_error(self, abp, pred):
        fs=125
        windowsize=1
        ma_perc=20
        pad_width = 19

        series = {}
        for label, windows in {'abp': abp, 'pred': pred}.items():
            sbp, dbp = [], []
            for x in tqdm(windows):
                x_pad = np.pad(x, pad_width=[pad_width, 0], constant_values=[x[0]])
                x_pad = np.pad(x_pad, pad_width=[0, pad_width], constant_values=[x[-1]])

                rol_mean = rolling_mean(x_pad, windowsize=windowsize, sample_rate=fs)
                peaks = detect_peaks(x_pad, rol_mean, ma_perc=ma_perc, sample_rate=fs)['peaklist']
                peaks = np.array(peaks) - pad_width - 1

                flip = flip_signal(x_pad)
                rol_mean = rolling_mean(flip, windowsize=windowsize, sample_rate=fs)
                valleys = detect_peaks(flip, rol_mean, ma_perc=ma_perc, sample_rate=fs)['peaklist']
                valleys = np.array(valleys) - pad_width - 1

                peak_mean = np.mean(x[peaks]) if len(peaks) > 0 else -1
                valley_mean = np.mean(x[valleys]) if len(valleys) > 0 else -1
                sbp.append(peak_mean)
                dbp.append(valley_mean)
            series[f'{label}_sbp'] = sbp
            series[f'{label}_dbp'] = dbp

        df = pd.DataFrame(series)
        df['sbp_err'] = df['pred_sbp'] - df['abp_sbp']
        df['dbp_err'] = df['pred_dbp'] - df['abp_dbp']
        df['sbp_abs_err'] = np.abs(df['sbp_err'])
        df['dbp_abs_err'] = np.abs(df['dbp_err'])

        for err, label in zip([df['sbp_abs_err'], df['dbp_abs_err']], ['Systolic BP', 'Diastolic BP']):
            print(label)
            print(f'{len(err[err < 15]) / len(err) * 100:.{3}}% < 15mmHg')
            print(f'{len(err[err < 10]) / len(err) * 100:.{3}}% < 10mmHg')
            print(f'{len(err[err < 5]) / len(err) * 100:.{3}}% < 5mmHg')
            print()
        return df
