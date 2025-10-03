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
polar code against an uncoded BPSK transmission over an AWGN channel.

Run the simulation with:

```
python crc_polar_vs_uncoded.py
```

Configuration is done by editing the `CONFIG` object near the bottom of the
script. There you can adjust the block length, CRC length/polynomial, list size,
SNR sweep, stopping criteria, RNG seed, and plotting preferences without
touching any command-line flags. The `CONFIG.min_frames_per_snr` and
`CONFIG.stop_when_error_free` options help accelerate high-SNR points by
terminating once no frame errors have been observed for the configured minimum
number of frames. By default the script evaluates SNR points from -2 dB to 6 dB
in 0.5 dB increments, prints a BER/FER summary table for both schemes, and
renders semi-log BER/FER curves when `matplotlib` is available. Set
`CONFIG.plot_results = False` if you prefer to skip figure generation or provide
`CONFIG.plot_file` to save the chart to disk instead of displaying it.

## CRC-polar over OFDM with LS channel estimation

`crc_polar_ofdm_ls.py` builds on the CRC-polar encoder/decoder together with the
least-squares OFDM channel estimator to compare performance against an ideal
receiver with perfect channel knowledge. Edit the in-file `CONFIG` object to set
the polar-code parameters, OFDM layout (subcarriers, pilot spacing, number of
OFDM symbols per frame, channel taps), and the SNR sweep. The
`CONFIG.min_frames_per_snr` and `CONFIG.stop_when_error_free` knobs let you exit
early at high SNR once no frame errors have been observed for the chosen number
of frames. Running

```
python crc_polar_ofdm_ls.py
```

prints a table that includes BER/FER for the LS-estimated receiver, the perfect
CSI baseline, and the average channel-estimation MSE accumulated across the
simulation. When `matplotlib` is available the script also renders a semi-log
plot of the BER/FER curves for both receivers; set `CONFIG.plot_results = False`
to skip figure generation or `CONFIG.plot_file = "ofdm_ls.png"` to save the
chart instead of displaying it.
