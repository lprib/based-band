# General structure

Order of operations:
- symbol dibits
- zero-interpolate up to BB sample rate
- pulse-shaping filter (RRC)
- multiply by frequency scalar
- frequency mod (actually phase mod with accumulator)
  - This produces complex IQ
- zero-order-hold interpolate up to passband sample rate
  - This is still complex IQ
- convolve with baseband-interolation-filter (runs in passband sample rate)
- modulate up to the passband channel
  - Turns IQ into a real-valued passband signal
- Pass through channel
- demodulate down to baseband
  - produces baseband IQ, but still at the passband sample rate
- Apply RX channel filter (low pass filter to remove adjacent channels), operates on IQ
- Downsample down the the baseband sample rate
- calculate phase of incoming signal (FM Modulated)
- Use adjacent-sample differences to get the "frequency/time" signal in BB
- matched pulse-shaping filter
- decision device (TODO) to get the symbols back

# TODO
- Why is the recieved demod/pulses different amplitudes?
- Can we just run everything at the passband sample rate (easier)
- CDMA