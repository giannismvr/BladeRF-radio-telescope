arxika: git clone https://github.com/Nuand/bladeRF.git

Ypotithetai pws akolouthw ayto to readme alla ta leei xalia-einai axrhstos: https://github.com/Nuand/bladeRF

install libusb

install cmake version > 3.5 and apply that version globally, for the whole computer

pip install lncurses or sth like that, i dont remember whether i installed it via pip

cd to the "bladeRF" folder

Then:

```
mkdir -p build
cd build
cmake .. -DCMAKE_EXE_LINKER_FLAGS="-lncurses"
make -j$(sysctl -n hw.ncpu)
sudo make install


```

now, assuming you are using a python venv, navigate to:
/Users/giannis/PycharmProjects/final_radiotelescope/bladeRF/bladeRF/host/libraries/libbladeRF_bindings/python

and then do:
python setup.py install