from . import hinerv
from . import matnerv2
from . import matnerv3
from . import matnerv4
from . import matnerv10
from . import matnerv20
from . import matnerv21
from . import mfnerv
from . import mgnerv

#
# Set the model classes here
#
model_cls = {
    'HiNeRV': hinerv,
    'MATNeRV2': matnerv2,
    'MATNeRV3': matnerv3,
    'MATNeRV4': matnerv4,
    'MATNeRV10': matnerv10,
    'MATNeRV20': matnerv20,
    'MATNeRV21': matnerv21,
    'MFNeRV': mfnerv,
    'MGNeRV': mgnerv,
}