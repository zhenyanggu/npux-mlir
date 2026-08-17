if {$argc != 1} {
  puts stderr "usage: xsct -nodisp probe_xsa.tcl <hardware.xsa>"
  exit 2
}

set hwDesign [lindex $argv 0]
hsi::open_hw_design $hwDesign
puts "hardware=[hsi::current_hw_design]"
puts "processors=[hsi::get_cells -filter {IP_TYPE == PROCESSOR}]"
hsi::close_hw_design [hsi::current_hw_design]
