#!/bin/sh
julia time_plot.jl results/12all0.5c0.5sserial.dat
julia time_plot.jl results/12all1.5c0.5sserial.dat
julia time_plot.jl results/12all0.5c2.0sserial.dat
julia time_plot.jl results/12all1.5c2.0sserial.dat
julia time_plot.jl results/12all0.5c0.5swrongmuserial.dat
julia time_plot.jl results/12all1.5c0.5swrongmuserial.dat
julia time_plot.jl results/12all0.5c2.0swrongmuserial.dat
julia time_plot.jl results/12all1.5c2.0swrongmuserial.dat
