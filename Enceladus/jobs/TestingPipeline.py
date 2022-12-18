import wandb
import keras
import numpy as np
import pandas as pd
import pickle as pkl
from tqdm import tqdm
import tensorflow as tf
from heartpy.preprocessing import flip_signal
from heartpy.peakdetection import detect_peaks
from heartpy.datautils import rolling_mean
from database_tools.tools import RecordsHandler


class TestingPipeline():
    def __init__(
        self,
        data_path,
        scaler_path,
        scaler_type,
        model_path,
    ):
        self._data_path = data_path
        self._scaler_path = scaler_path
        self._scaler_type = scaler_type
        self._model_path = model_path

    def run(self):
        print('Loading data...')
        handler = RecordsHandler(data_dir=self._data_path)
        test = handler.read_records(['test'], ['ppg', 'vpg', 'apg', 'abp'], n_cores=10, AUTOTUNE=tf.data.AUTOTUNE)['test']
        ppg, vpg, apg, abp = self._load_data(test)

        model = self._load_model(self._model_path)

        print('Generating predictions...')
        pred = model.predict([ppg, vpg, apg]).reshape(-1, 256)

        print('Rescaling data...')
        scaled = self._scale_data(self._scaler_path, self._scaler_type, ppg, vpg, apg, abp, pred)

        abp_scaled = scaled['abp']
        pred_scaled = scaled['pred']

        print('Calculating error...')
        self._calculate_error(abp_scaled, pred_scaled)
        return

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

    def _load_model(self, path):
        run = wandb.init()
        artifact = run.use_artifact(path, type='model')
        artifact_dir = artifact.download()
        wandb.finish()
        model = keras.models.load_model(artifact_dir)
        return model

    def _scale_data(self, path, scaler_type, ppg, vpg, apg, abp, pred):
        with open(path, 'rb') as f:
            scalers = pkl.load(f)

        ppg_scaler = scalers['ppg']
        vpg_scaler = scalers['vpg']
        apg_scaler = scalers['apg']
        abp_scaler = scalers['abp']

        if scaler_type == 'MinMax':
            ppg = np.multiply(ppg, ppg_scaler[1] - ppg_scaler[0]) + ppg_scaler[0]
            vpg = np.multiply(vpg, vpg_scaler[1] - vpg_scaler[0]) + vpg_scaler[0]
            apg = np.multiply(apg, apg_scaler[1] - apg_scaler[0]) + apg_scaler[0]
            abp = np.multiply(abp, abp_scaler[1] - abp_scaler[0]) + abp_scaler[0]
            pred = np.multiply(pred, abp_scaler[1] - abp_scaler[0]) + abp_scaler[0]
        elif scaler_type == 'Standard':
            pass
        scaled_data = dict(ppg=ppg, vpg=vpg, apg=apg, abp=abp, pred=pred)
        model_name = self._model_path.split('/')[-1]
        with open(f'{model_name}_predictions.pkl', 'wb') as f:
            pkl.dump(scaled_data, f)
        return scaled_data

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

        model_name = self._model_path.split('/')[-1]
        with open(f'{model_name}_error.pkl', 'wb') as f:
            pkl.dump(df, f)

        for err, label in zip([df['sbp_abs_err'], df['dbp_abs_err']], ['Systolic BP', 'Diastolic BP']):
            print(label)
            print(f'{len(err[err < 15]) / len(err) * 100:.{3}}% < 15mmHg')
            print(f'{len(err[err < 10]) / len(err) * 100:.{3}}% < 10mmHg')
            print(f'{len(err[err < 5]) / len(err) * 100:.{3}}% < 5mmHg')
            print()
        return
