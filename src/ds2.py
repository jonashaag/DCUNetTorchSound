from torch.utils.data import DataLoader, Dataset
import os, glob, ujson, random, librosa, scipy.signal, pyloudnorm, numpy as np, pickle, tqdm, tqdm.contrib.concurrent
import hashlib

np.seterr("raise")


def trim_zeros_block(filt, trim='fb', block_size=1024):
    """Trim blocks of zeros"""
    trim = trim.upper()
    first = 0
    if 'F' in trim:
        for i in range(0, len(filt), block_size):
            if np.any(filt[i:i+block_size] != 0.):
                first = i
                break
    last = len(filt)
    if 'B' in trim:
        for i in range(len(filt)-1, block_size - 1, -block_size):
            if np.any(filt[i-block_size:i] != 0.):
                last = i
                break
    return filt[first:last]


def fast_trim_zeros(filt, trim='fb'):
    filt = trim_zeros_block(filt, trim)
    return np.trim_zeros(filt, trim)



def deterministic_shuffle(x):
    random.Random(4242).shuffle(x)


def deterministic_sample(x, n):
    return random.Random(4242).sample(x, n)


def json_hash(*xs):
    s = ujson.dumps(
        [deterministic_sample(x, min(10_000, int(len(x) / 10))) for x in xs]
    )
    return hashlib.md5(s.encode("utf8")).hexdigest()


def loadmono(f, expected_sr, **kw):
    data, sr = librosa.core.load(f, sr=None, mono=False, **kw)
    assert sr == expected_sr, (f, expected_sr, sr)
    assert data.ndim == 1, f"{f} is not mono"
    return data


class ListDataset(Dataset):
    def __init__(self, x):
        self.x = x

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx]


class MyFastDataset(Dataset):
    def __init__(self, sr, segment, clean_files, ir_files, ir_duration, n):
        self.sr = sr
        self.segment = segment
        self.clean_files = clean_files[:]
        self.ir_files = ir_files[:]
        self.ir_duration = ir_duration
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        # e.g 2s segment
        chunk_len = int(self.segment * self.sr)
        # eg. 2s segment + 1s IR
        source_chunk_len = chunk_len + int(self.ir_duration * self.sr)

        # Get random IR
        ir, ir_f, ir_idx = self.find_ir()

        # Get random clean
        clean, clean_f, clean_idx = self.find_clean(source_chunk_len)
        random_start = np.random.randint(0, len(clean) + 1 - source_chunk_len)
        clean_chunk = clean[random_start:][:source_chunk_len]

        # Convolve
        direct_frames = int(ir.argmax() + self.sr * (0.01 + 0.001 * (ir_idx % 10)))
        direct_ir = librosa.util.fix_length(ir[:direct_frames], len(ir))

        try:
            direct = pyloudnorm.normalize.peak(
                scipy.signal.convolve(clean_chunk, direct_ir, "valid")[1:], -1
            )
            reverberant = pyloudnorm.normalize.peak(
                scipy.signal.convolve(clean_chunk, ir, "valid")[1:], -1
            )
        except Exception as err:
            raise RuntimeError(f"Error with {clean_f} {ir_f} {random_start} {direct_frames}") from err
        assert direct.shape == reverberant.shape

        # Transform for Asteroid
        sources = librosa.util.fix_length(direct, chunk_len)
        noisy = librosa.util.fix_length(reverberant, chunk_len)
        return noisy, sources

    def find_clean(self, min_len):
        """Find suitable clean"""
        while True:
            clean_idx = np.random.randint(0, len(self.clean_files))
            clean_f = self.clean_files[clean_idx]
            if clean_f is None:
                # we already removed this in the past
                continue
            clean = loadmono(clean_f, self.sr)
            if len(clean) < min_len:
                # File too short
                self.clean_files[clean_idx] = None
                continue
            return clean, clean_f, clean_idx

    def find_ir(self):
        """Find suitable IR"""
        while True:
            ir_idx = np.random.randint(0, len(self.ir_files))
            ir_f = self.ir_files[ir_idx]
            if ir_f is None:
                # we already removed this in the past
                continue
            ir = loadmono(ir_f, self.sr)
            ir = fast_trim_zeros(ir)
            # TODO: find better way to deal with these (late argmax) (0.45 - was 1770)
            if not ir.size or ir.argmax() >= 0.45 * self.ir_duration * self.sr:
                self.ir_files[ir_idx] = None
                continue
            return ir[:int(self.ir_duration * self.sr)], ir_f, ir_idx


def groupby(it, key):
    bykey = {}
    for x in it:
        bykey.setdefault(key(x), []).append(x)
    return bykey


def make_split(it, key, splits):
    assert 0.999 <= sum(splits) <= 1.001

    it = it[:]
    it.sort()
    deterministic_shuffle(it)

    bykey = groupby(it, key)
    keys_list = list(bykey.keys())
    keys_list.sort()
    deterministic_shuffle(keys_list)

    splits = [int(len(keys_list) * r) for r in splits[:-1]] + [len(keys_list)]
    for nkeys in splits:
        this_keys = keys_list[:nkeys]
        yield this_keys[:], sum([bykey[k] for k in this_keys], [])
        keys_list = keys_list[nkeys:]


def getds(for_test, conf):
    assert not for_test

    if 1:
        (
            (train_spks, train_clean),
            (val_spks, val_clean),
            (test_spks, test_clean),
        ) = make_split(
            glob.glob(conf["data"]["clean_files"], recursive=True),
            lambda f: f.split("_reader_" if "_reader_" in f else "/p")[1].split("_")[0],
            [0.85, 0.1, 0.05],
        )

        (_, train_irs), (_, val_irs), (_, test_irs) = make_split(
            glob.glob(conf["data"]["ir_files"], recursive=True),
            lambda x: x,
            [0.85, 0.1, 0.05],
        )

        print(
            "Speech",
            len(train_spks),
            len(val_spks),
            len(test_spks),
            len(train_clean),
            len(val_clean),
            len(test_clean),
            json_hash(train_spks),
            json_hash(val_spks),
            json_hash(test_spks),
        )
        print(
            "IR",
            len(train_irs),
            len(val_irs),
            len(test_irs),
            json_hash(train_irs),
            json_hash(val_irs),
            json_hash(test_irs),
        )

        train_set = MyFastDataset(
            conf["data"]["sample_rate"],
            conf["data"]["segment"],
            train_clean,
            train_irs,
            ir_duration=1.0,
            #n=100_000,
            n=10_000,
        )

        # 1_000 pcs ~ 0.1G mem
        val_set = MyFastDataset(
            conf["data"]["sample_rate"],
            conf["data"]["segment"],
            val_clean,
            val_irs,
            ir_duration=1.0,
            #n=10_000,
            n=3_000,
        )
        val_set = ListDataset(
            tqdm.contrib.concurrent.thread_map(lambda _: train_set[0], range(len(val_set))))

    else:
        clean = [
            x for x in
            #( len(librosa.core.load(x, sr=None)[0])/16000) for x in
            deterministic_sample(sorted(glob.glob(conf["data"]["clean_files"], recursive=True)), 50)
            if len(librosa.core.load(x, sr=None)[0])/16000 > 4
        ]
        train_set = MyFastDataset(
            conf["data"]["sample_rate"],
            conf["data"]["segment"],
            clean[:5],
            #deterministic_sample(sorted(glob.glob(conf["data"]["ir_files"], recursive=True)), 3),
            glob.glob(conf["data"]["ir_files"], recursive=True),
            ir_duration=1.0,
            n=15_000)

        val_set = MyFastDataset(
            conf["data"]["sample_rate"],
            conf["data"]["segment"],
            deterministic_sample(sorted(glob.glob(conf["data"]["clean_files"], recursive=True)), 5),
            deterministic_sample(sorted(glob.glob(conf["data"]["ir_files"], recursive=True)), 200),
            ir_duration=1.0,
            n=1_000)


    train_loader = DataLoader(
        train_set,
        shuffle=False,
        batch_size=conf["training"]["batch_size"],
        num_workers=conf["training"]["num_workers"],
        drop_last=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        shuffle=False,
        batch_size=conf["training"]["batch_size"],
        num_workers=conf["training"]["num_workers"],
        drop_last=True,
        pin_memory=True,
    )

    return train_loader, val_loader
