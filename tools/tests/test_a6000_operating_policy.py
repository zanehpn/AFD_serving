import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from a6000_operating_policy import check_operating_point

CONFIG={'expected_gpu_model':'NVIDIA RTX A6000','memory_clock_mhz':8001,'memory_clock_policy':'driver_managed_pstates'}
CANDIDATE={'attention_gpus':[0,1],'attention_power_w':300,'expert_power_w':250}

@pytest.mark.parametrize('memory',[810,5001,7601,8001])
def test_driver_pstates_keep_power_validation(memory):
    state={'power_limit_w':300,'memory_clock_mhz':memory}
    check_operating_point(CONFIG,CANDIDATE,0,state)
    with pytest.raises(ValueError,match='power cap'):
        check_operating_point(CONFIG,CANDIDATE,2,state)

@pytest.mark.parametrize('memory',[0,9000,float('nan')])
def test_bad_telemetry_rejected(memory):
    with pytest.raises(ValueError,match='Invalid'):
        check_operating_point(CONFIG,CANDIDATE,0,{'power_limit_w':300,'memory_clock_mhz':memory})

def test_historical_fixed_clock_contract_preserved():
    with pytest.raises(ValueError,match='memory clock differs'):
        check_operating_point({'memory_clock_mhz':1215},CANDIDATE,0,{'power_limit_w':300,'memory_clock_mhz':810})
    check_operating_point({'memory_clock_mhz':1215},CANDIDATE,0,{'power_limit_w':300,'memory_clock_mhz':1215})
