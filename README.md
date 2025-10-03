# List Decoder for Polar Codes, CRC-Polar Codes, and PAC Codes
If you find this algorithm useful, please cite the following paper. Thanks.

M. Rowshan, A. Burg and E. Viterbo, "Polarization-Adjusted Convolutional (PAC) Codes: Sequential Decoding vs List Decoding," in IEEE Transactions on Vehicular Technology, vol. 70, no. 2, pp. 1434-1447, Feb. 2021, doi: 10.1109/TVT.2021.3052550.

https://ieeexplore.ieee.org/abstract/document/9328621

Description: 
This is an implementation of the successive cancellation list (SCL) decoding algorithm for polar codes, CRC-polar codes, and PAC codes with the choice of various code constructions/rate-profiles in Python. 
The list decoding algorithm is an adaptive two stage successive cancellation list (SCL) algorithm. That means first it tries L=1 and then L=L_max. The performance is the same as list decoding with L_max. This trick has been implemented in the simulator.py file. The rest of the files are the same as the standard list decoding algorithm.

The main file is simulator.py in which you can set the parameters of the code and the channel.

To switch between decoding polar codes and PAC codes, you need to change the generator polynomial conv_gen to conv_gen=[1] for polar codes or any other polynomial such as conv_gen=[1,0,1,1,0,1,1].

## The differences between PAC codes and Polar codes are in 
- the encoding process where we have one more stage that we call convolutional precoding or pre-transformation. If conv_gen = [1], that means no precoding is performed, hence the result will be the simulation for polar codes, not PAC codes. If you look at  pac_encode() method in polar_code.py file, you would find the convolutional precoding method, conv_encode(), as U = pcfun.conv_encode(V, conv_gen, mem).
- the decoding process where we consider both 0 and 1 values for every v and obtain the corresponding u by conv_1bit() method based on the current state. Then, we calculate the path metric based on the LLR and these u values (the values 0 and 1 of each extended branch on the tree). Obviously, we need to update the current state as well by getNextState() method.

Note that the "copy on write" or "lazy copy" technique has been used in this algorithm.

Please report any bugs to mrowshan at ieee dot org

## CRC-polar vs. uncoded baseline simulation

The repository now includes a helper script, `crc_polar_vs_uncoded.py`, which
uses the existing encoder/decoder to compare the performance of a CRC-aided
polar code against an uncoded BPSK transmission over an AWGN channel. Example
usage:

```
python crc_polar_vs_uncoded.py --n 128 --k-info 64 --crc-length 16 --list-size 16 \
    --target-frame-errors 30 --max-frames 5000 --seed 1 --plot-file results.png
```

By default the script evaluates SNR points from -2 dB to 6 dB in 0.5 dB
increments. It prints a table with BER/FER metrics for both schemes and also
plots semi-log curves for the coded and uncoded BER/FER. Use `--no-plot` if you
prefer to skip figure generation or `--plot-file` to save the chart to disk.
    --snr 0 1 2 3 --target-frame-errors 30 --max-frames 5000 --seed 1
```

The script prints a table with BER/FER metrics for both schemes across the
requested SNR points.
