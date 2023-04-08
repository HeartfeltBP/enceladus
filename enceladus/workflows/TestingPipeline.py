import glob
import pickle as pkl

import keras
import numpy as np
import pandas as pd
import wandb
from tqdm import tqdm

from database_tools.processing.detect import detect_peaks
from database_tools.tools.records import read_records, rescale_data


class TestingPipeline:
    def __init__(
        self,
        data_dir: str,
        model_dir: str,
    ):
        self.data_dir = data_dir
        self.model_dir = model_dir
        scaler_dir = glob.glob(data_dir[0:-5] + 'records_info_*.pkl')[0]
        with open(scaler_dir, 'rb') as f:
            self.scaler, self.split_idx = pkl.load(f)
        self.model = self._load_model(model_dir)

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
        # return (data, ppg, vpg, apg, abp, self.model)

        print('Generating predictions...')
        pred = self.model.predict([ppg, vpg, apg]).reshape(-1, 256)
        # return pred

        print('Rescaling data...')
        ppg_scaled = rescale_data(ppg, self.scaler['ppg'])
        vpg_scaled = rescale_data(vpg, self.scaler['vpg'])
        apg_scaled = rescale_data(apg, self.scaler['apg'])
        # abp_scaled = rescale_data(abp, self.scaler['abp'])
        # pred_scaled = rescale_data(pred, self.scaler['abp'])
        # return (ppg_scaled, vpg_scaled, apg_scaled, abp_scaled, pred_scaled)

        print('Calculating error...')
        df = self._calculate_error(abp, pred)
        return df, ppg_scaled, vpg_scaled, apg_scaled, abp, pred

    def _load_data(self, data):
        ppg, vpg, apg, abp = [], [], [], []
        for i, (inputs, label) in tqdm(enumerate(data.as_numpy_iterator())):
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
        series = {}
        for label, windows in {'abp': abp, 'pred': pred}.items():
            sbp, dbp = [], []
            for x in tqdm(windows):
                pad_width = 40
                x_pad = np.pad(x, pad_width=pad_width, constant_values=np.mean(x))

                peaks, troughs = detect_peaks(x_pad).values()
                peaks = peaks - pad_width - 1
                troughs = troughs - pad_width - 1

                peak_mean = np.mean(x[peaks]) if len(peaks) > 0 else -1
                valley_mean = np.mean(x[troughs]) if len(troughs) > 0 else -1

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
