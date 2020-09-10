#! /usr/bin/env bash

cd ..
# CLI tools
## fcs_convert
## Conversion to Kristine
### Kristine -> Kristine
python chisurf/cmd_tools/fcs_convert.py -if "./test/data/fcs/kristine/Kristine_without_error.cor" -it kristine -of test.cor -ot kristine
### ALV/ASC -> Kristine
python chisurf/cmd_tools/fcs_convert.py -if "./test/data/fcs/asc/ALV-5000E-WIN.ASC" -it alv -of test.cor -ot kristine
python chisurf/cmd_tools/fcs_convert.py -if "./test/data/fcs/asc/ALV-7004.ASC" -it alv -of test.cor -ot kristine
python chisurf/cmd_tools/fcs_convert.py -if "./test/data/fcs/asc/ALV-7004USB_dia10_cen10_0001.ASC" -it alv -of test.cor -ot kristine
python chisurf/cmd_tools/fcs_convert.py -if "./test/data/fcs/asc/ALV-7004USB_ac01_cc01_10.ASC" -it alv -of test.cor -ot kristine
python chisurf/cmd_tools/fcs_convert.py -if "./test/data/fcs/asc/ALV-7004USB_ac3.ASC" -it alv -of test.cor -ot kristine
### China mat/ -> Kristine
python chisurf/cmd_tools/fcs_convert.py -if "./test/data/fcs/china_mat/Alexa488.mat" -it china-mat -of test.cor -ot kristine
python chisurf/cmd_tools/fcs_convert.py -if "./test/data/fcs/china_mat/nano_beads.mat" -it china-mat -of test.cor -ot kristine
### confocor 3 -> Kristine
python chisurf/cmd_tools/fcs_convert.py -if "./test/data/fcs/confocor3/Zeiss_Confocor3_A488+GFP/001_A488.fcs" -it confocor3 -of test.cor -ot kristine
python chisurf/cmd_tools/fcs_convert.py -if "./test/data/fcs/confocor3/Zeiss_Confocor3_LSM780_FCCS_HeLa_2015/017_cp_KIND+BFA.fcs" -it confocor3 -of test.cor -ot kristine
### pycorrfit -> Kristine
python chisurf/cmd_tools/fcs_convert.py -if "./test/data/fcs/pycorrfit/PyCorrFit_CC_A488_withVariance.csv" -it pycorrfit -of test.cor -ot kristine
python chisurf/cmd_tools/fcs_convert.py -if "./test/data/fcs/pycorrfit/PyCorrFit_CC_A488_withTrace.csv" -it pycorrfit -of test.cor -ot kristine
python chisurf/cmd_tools/fcs_convert.py -if "./test/data/fcs/pycorrfit/PyCorrFit_CC_A488.csv" -it pycorrfit -of test.cor -ot kristine
### pq.dat -> Kristine
python chisurf/cmd_tools/fcs_convert.py -if "./test/data/fcs/pq.fcs/A488_3.dat" -it "pq.dat" -of test.cor -ot kristine
rm test*.cor

## Conversion to yaml
### Kristine -> yaml
python chisurf/cmd_tools/fcs_convert.py -if "./test/data/fcs/kristine/Kristine_without_error.cor" -it kristine -of test.yaml -ot yaml
### ALV/ASC -> yaml
python chisurf/cmd_tools/fcs_convert.py -if "./test/data/fcs/asc/ALV-5000E-WIN.ASC" -it alv -of test.yaml -ot yaml
python chisurf/cmd_tools/fcs_convert.py -if "./test/data/fcs/asc/ALV-7004.ASC" -it alv -of test.yaml -ot yaml
python chisurf/cmd_tools/fcs_convert.py -if "./test/data/fcs/asc/ALV-7004USB_dia10_cen10_0001.ASC" -it alv -of test.yaml -ot yaml
python chisurf/cmd_tools/fcs_convert.py -if "./test/data/fcs/asc/ALV-7004USB_ac01_cc01_10.ASC" -it alv -of test.yaml -ot yaml
python chisurf/cmd_tools/fcs_convert.py -if "./test/data/fcs/asc/ALV-7004USB_ac3.ASC" -it alv -of test.yaml -ot yaml
### China mat/ -> yaml
python chisurf/cmd_tools/fcs_convert.py -if "./test/data/fcs/china_mat/Alexa488.mat" -it china-mat -of test.yaml -ot yaml
python chisurf/cmd_tools/fcs_convert.py -if "./test/data/fcs/china_mat/nano_beads.mat" -it china-mat -of test.yaml -ot yaml
### confocor 3 -> yaml
python chisurf/cmd_tools/fcs_convert.py -if "./test/data/fcs/confocor3/Zeiss_Confocor3_A488+GFP/001_A488.fcs" -it confocor3 -of test.yaml -ot yaml
python chisurf/cmd_tools/fcs_convert.py -if "./test/data/fcs/confocor3/Zeiss_Confocor3_LSM780_FCCS_HeLa_2015/017_cp_KIND+BFA.fcs" -it confocor3 -of test.yaml -ot yaml
### pycorrfit -> yaml
python chisurf/cmd_tools/fcs_convert.py -if "./test/data/fcs/pycorrfit/PyCorrFit_CC_A488_withVariance.csv" -it pycorrfit -of test.yaml -ot yaml
python chisurf/cmd_tools/fcs_convert.py -if "./test/data/fcs/pycorrfit/PyCorrFit_CC_A488_withTrace.csv" -it pycorrfit -of test.yaml -ot yaml
python chisurf/cmd_tools/fcs_convert.py -if "./test/data/fcs/pycorrfit/PyCorrFit_CC_A488.csv" -it pycorrfit -of test.yaml -ot yaml
### pq.dat -> yaml
python chisurf/cmd_tools/fcs_convert.py -if "./test/data/fcs/pq.fcs/A488_3.dat" -it "pq.dat" -of test.yaml -ot yaml
rm test.yaml
