import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from safetensors.numpy import save_file

from tools.model.flash_teacher_inputs import Inputs
from tools.quantization.flash_next import digest


class InputTests(unittest.TestCase):
    def fixture(self,root):
        contract={'source':'Qwen/original','revision':'0'*40,'config_sha256':'config',
                  'scenes_sha256':'scenes','index_sha256':'index','seed':20261002,
                  'format':'orinfer.original-bf16-inputs.v1'}
        values=np.array([[1.,-.5],[.25,2.]],dtype=np.float32)
        save_file({'embedding.x':values,'ple.x':values},str(root/'inputs.safetensors'))
        manifest={'contract':contract,'sha256':digest(root/'inputs.safetensors'),
                  'tokens':{'x':[1,2]},'source_ranges':[],'complete':True}
        (root/'inputs.json').write_text(json.dumps(manifest))
        identity={k:v for k,v in contract.items() if k not in ('format','seed')}
        return values,manifest,identity

    def test_source_history_and_exact_bf16_value_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);values,manifest,identity=self.fixture(root)
            loaded=Inputs(root,**identity)
            case={'id':'x','prompt_ids':[1],'target_ids':[2,3]}
            for value in loaded.case(case,2):np.testing.assert_array_equal(value,values)
            with self.assertRaises(ValueError):loaded.case(dict(case,prompt_ids=[4]),2)
            with self.assertRaises(ValueError):Inputs(root,**dict(identity,revision='1'*40))
            changed=values.copy();changed[0,0]=np.nextafter(changed[0,0],np.float32(2.))
            save_file({'embedding.x':changed,'ple.x':values},str(root/'inputs.safetensors'))
            with self.assertRaises(ValueError):Inputs(root,**identity)
            manifest['sha256']=digest(root/'inputs.safetensors');(root/'inputs.json').write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):Inputs(root,**identity).case(case,2)


if __name__ == '__main__':unittest.main()
