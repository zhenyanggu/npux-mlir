# Generate the ZCU102 Linux runtime FPGA overlay source.
#
# The base system is the Xilinx-published Ubuntu image.  Its boot-time DT must
# expose the zynqmp FPGA manager (label: zynqmp_fpga) and the GIC (label: gic).
# Compile the emitted source with dtc -@; do not use PetaLinux/XSA generation.
#
# Usage:
#   vivado -mode batch -source dt_overlay.tcl -tclargs \
#       --output /tmp/npu.dtso --firmware-name versaedge_npu.bit.bin

proc usage {} {
    puts "usage: dt_overlay.tcl --output <file.dtso> --firmware-name <bit.bin> ?--irq <GIC_SPI>? ?--dma-buffer-size <bytes>? ?--target-path </amba>? ?--no-irq?"
}
proc require_value {argv_var index_var option} {
    upvar 1 $argv_var argv $index_var index
    incr index
    if {$index >= [llength $argv]} { error "Missing value for $option" }
    return [lindex $argv $index]
}

set output ""
set firmware_name ""
set irq 121
set dma_buffer_size 0x10000000
set target_path "/amba"
set include_irq 1
set index 0
while {$index < [llength $argv]} {
    set option [lindex $argv $index]
    switch -- $option {
        --output { set output [file normalize [require_value argv index $option]] }
        --firmware-name { set firmware_name [require_value argv index $option] }
        --irq { set irq [require_value argv index $option] }
        --dma-buffer-size { set dma_buffer_size [require_value argv index $option] }
        --target-path { set target_path [require_value argv index $option] }
        --no-irq { set include_irq 0 }
        --help { usage; return }
        default { usage; error "Unknown option: $option" }
    }
    incr index
}
if {$output eq "" || $firmware_name eq ""} {
    usage
    error "--output and --firmware-name are required"
}
if {![string is integer -strict $irq] || $irq < 0} {
    error "--irq must be a non-negative GIC SPI number"
}
if {![string is integer -strict $dma_buffer_size] || $dma_buffer_size <= 0 || ($dma_buffer_size % 4096) != 0} {
    error "--dma-buffer-size must be a positive 4096-byte aligned integer"
}
if {![string match "/*" $target_path]} {
    error "--target-path must be an absolute device-tree path"
}

set fd [open $output w]
puts $fd "/dts-v1/;"
puts $fd "/plugin/;"
puts $fd ""
puts $fd "/ {"
puts $fd "    compatible = \"xlnx,zynqmp-zcu102\";"
puts $fd ""
puts $fd "    fragment@0 {"
puts $fd "        target-path = \"/\";"
puts $fd "        __overlay__ {"
puts $fd "            npu_fpga_region: npu-fpga-region {"
puts $fd "                compatible = \"fpga-region\";"
puts $fd "                fpga-mgr = <&zynqmp_fpga>;"
puts $fd "                firmware-name = \"$firmware_name\";"
puts $fd "            };"
puts $fd "        };"
puts $fd "    };"
puts $fd ""
puts $fd "    fragment@1 {"
puts $fd "        target-path = \"$target_path\";"
puts $fd "        __overlay__ {"
puts $fd "            npu@a0000000 {"
puts $fd "                compatible = \"xlnx,versaedge-npu-zcu102-1.0\";"
puts $fd "                reg = <0x0 0xa0000000 0x0 0x00010000>;"
puts $fd "                xlnx,dma-buffer-size = <0x[format %08x $dma_buffer_size]>;"
if {$include_irq} {
    puts $fd "                interrupt-parent = <&gic>;"
    puts $fd "                interrupts = <0 $irq 4>;"
}
puts $fd "            };"
puts $fd "        };"
puts $fd "    };"
puts $fd "};"
close $fd
puts "DT_OVERLAY_SOURCE=$output"
puts "CONTROL_BASE=0xA0000000"
puts "GIC_SPI=$irq"
puts "DMA_BUFFER_SIZE=$dma_buffer_size"
