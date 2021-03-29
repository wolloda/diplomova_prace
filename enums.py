from enum import Enum

class DatasetDirs(Enum):
    COPHIR_1M = "/storage/brno6/home/tslaninakova/learned-indexes/MTree1M"
    COPHIR_100k = "/storage/brno6/home/tslaninakova/learned-indexes/MTree100k"
    PROFI_1M = "/storage/brno6/home/tslaninakova/learned-indexes/MtreeProfi2000"
    PROFI_100k = "/storage/brno6/home/tslaninakova/learned-indexes/MtreeProfi200"
    MOCAP = "/storage/brno6/home/tslaninakova/learned-indexes/MTree1M-mocap"

DEFAULT_DESCRIPTORS = {
        DatasetDirs.COPHIR_100k: 282,
        DatasetDirs.COPHIR_1M: 282,
        DatasetDirs.PROFI_100k: 4096,
        DatasetDirs.PROFI_1M: 4096,
        DatasetDirs.MOCAP: 4096
}

KNN_OBJECTS = {
    DatasetDirs.COPHIR_100k: "/storage/brno6/home/tslaninakova/learned-indexes/datasets/queries.data",
    DatasetDirs.COPHIR_1M: "/storage/brno6/home/tslaninakova/learned-indexes/datasets/queries.data",
    DatasetDirs.PROFI_100k: "/storage/brno6/home/tslaninakova/learned-indexes/datasets/profiset-queries.data",
    DatasetDirs.PROFI_1M: "/storage/brno6/home/tslaninakova/learned-indexes/datasets/profiset-queries.data",
    DatasetDirs.MOCAP: "/storage/brno6/home/tslaninakova/learned-indexes/datasets/mocap-queries.data"
}
