if {$argc != 2} {
  puts stderr "usage: xsct -nodisp build_zynqmp_fsbl.tcl <hardware.xsa> <workspace>"
  exit 2
}

set hwDesign [file normalize [lindex $argv 0]]
set workspace [file normalize [lindex $argv 1]]

setws $workspace
platform create -name zcu102_npu_platform -hw $hwDesign -proc psu_cortexa53_0 -os standalone
platform write
platform generate
app create -name zcu102_npu_fsbl -platform zcu102_npu_platform -domain standalone_domain -template {Zynq MP FSBL}
app build -name zcu102_npu_fsbl
