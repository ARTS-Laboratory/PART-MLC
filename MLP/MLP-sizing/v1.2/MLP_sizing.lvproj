<?xml version='1.0' encoding='UTF-8'?>
<Project Type="Project" LVVersion="20008000">
	<Item Name="My Computer" Type="My Computer">
		<Property Name="server.app.propertiesEnabled" Type="Bool">true</Property>
		<Property Name="server.control.propertiesEnabled" Type="Bool">true</Property>
		<Property Name="server.tcp.enabled" Type="Bool">false</Property>
		<Property Name="server.tcp.port" Type="Int">0</Property>
		<Property Name="server.tcp.serviceName" Type="Str">My Computer/VI Server</Property>
		<Property Name="server.tcp.serviceName.default" Type="Str">My Computer/VI Server</Property>
		<Property Name="server.vi.callsEnabled" Type="Bool">true</Property>
		<Property Name="server.vi.propertiesEnabled" Type="Bool">true</Property>
		<Property Name="specify.custom.address" Type="Bool">false</Property>
		<Item Name="host.vi" Type="VI" URL="../host.vi"/>
		<Item Name="memory_load_B_0.vi" Type="VI" URL="../memory_load_B_0.vi"/>
		<Item Name="memory_load_B_1.vi" Type="VI" URL="../memory_load_B_1.vi"/>
		<Item Name="memory_load_B_2.vi" Type="VI" URL="../memory_load_B_2.vi"/>
		<Item Name="memory_load_C_0.vi" Type="VI" URL="../memory_load_C_0.vi"/>
		<Item Name="memory_load_C_1.vi" Type="VI" URL="../memory_load_C_1.vi"/>
		<Item Name="memory_load_C_2.vi" Type="VI" URL="../memory_load_C_2.vi"/>
		<Item Name="Dependencies" Type="Dependencies">
			<Item Name="vi.lib" Type="Folder">
				<Item Name="Application Directory.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/file.llb/Application Directory.vi"/>
				<Item Name="Error Cluster From Error Code.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/Error Cluster From Error Code.vi"/>
				<Item Name="NI_AALBase.lvlib" Type="Library" URL="/&lt;vilib&gt;/Analysis/NI_AALBase.lvlib"/>
				<Item Name="NI_FileType.lvlib" Type="Library" URL="/&lt;vilib&gt;/Utility/lvfile.llb/NI_FileType.lvlib"/>
				<Item Name="NI_Matrix.lvlib" Type="Library" URL="/&lt;vilib&gt;/Analysis/Matrix/NI_Matrix.lvlib"/>
			</Item>
			<Item Name="lvanlys.dll" Type="Document" URL="/&lt;resource&gt;/lvanlys.dll"/>
			<Item Name="NiFpgaLv.dll" Type="Document" URL="NiFpgaLv.dll">
				<Property Name="NI.PreserveRelativePath" Type="Bool">true</Property>
			</Item>
			<Item Name="subvi_load_data_matrix.vi" Type="VI" URL="../subvi_load_data_matrix.vi"/>
			<Item Name="subvi_load_data_vector.vi" Type="VI" URL="../subvi_load_data_vector.vi"/>
		</Item>
		<Item Name="Build Specifications" Type="Build"/>
	</Item>
	<Item Name="NI-cRIO-9054-01F17EC3" Type="RT CompactRIO">
		<Property Name="alias.name" Type="Str">NI-cRIO-9054-01F17EC3</Property>
		<Property Name="alias.value" Type="Str">172.22.11.2</Property>
		<Property Name="CCSymbols" Type="Str">TARGET_TYPE,RT;OS,Linux;CPU,x64;DeviceCode,79DE;</Property>
		<Property Name="crio.ControllerPID" Type="Str">79DE</Property>
		<Property Name="host.ResponsivenessCheckEnabled" Type="Bool">true</Property>
		<Property Name="host.ResponsivenessCheckPingDelay" Type="UInt">5000</Property>
		<Property Name="host.ResponsivenessCheckPingTimeout" Type="UInt">1000</Property>
		<Property Name="host.TargetCPUID" Type="UInt">9</Property>
		<Property Name="host.TargetOSID" Type="UInt">19</Property>
		<Property Name="host.TargetUIEnabled" Type="Bool">false</Property>
		<Property Name="target.cleanupVisa" Type="Bool">false</Property>
		<Property Name="target.FPProtocolGlobals_ControlTimeLimit" Type="Int">300</Property>
		<Property Name="target.getDefault-&gt;WebServer.Port" Type="Int">80</Property>
		<Property Name="target.getDefault-&gt;WebServer.Timeout" Type="Int">60</Property>
		<Property Name="target.IOScan.Faults" Type="Str"></Property>
		<Property Name="target.IOScan.NetVarPeriod" Type="UInt">100</Property>
		<Property Name="target.IOScan.NetWatchdogEnabled" Type="Bool">false</Property>
		<Property Name="target.IOScan.Period" Type="UInt">10000</Property>
		<Property Name="target.IOScan.PowerupMode" Type="UInt">0</Property>
		<Property Name="target.IOScan.Priority" Type="UInt">0</Property>
		<Property Name="target.IOScan.ReportModeConflict" Type="Bool">true</Property>
		<Property Name="target.IsRemotePanelSupported" Type="Bool">true</Property>
		<Property Name="target.RTCPULoadMonitoringEnabled" Type="Bool">true</Property>
		<Property Name="target.RTDebugWebServerHTTPPort" Type="Int">8001</Property>
		<Property Name="target.RTTarget.ApplicationPath" Type="Path">/c/ni-rt/startup/startup.rtexe</Property>
		<Property Name="target.RTTarget.EnableFileSharing" Type="Bool">true</Property>
		<Property Name="target.RTTarget.IPAccess" Type="Str">+*</Property>
		<Property Name="target.RTTarget.LaunchAppAtBoot" Type="Bool">false</Property>
		<Property Name="target.RTTarget.VIPath" Type="Path">/home/lvuser/natinst/bin</Property>
		<Property Name="target.server.app.propertiesEnabled" Type="Bool">true</Property>
		<Property Name="target.server.control.propertiesEnabled" Type="Bool">true</Property>
		<Property Name="target.server.tcp.access" Type="Str">+*</Property>
		<Property Name="target.server.tcp.enabled" Type="Bool">false</Property>
		<Property Name="target.server.tcp.paranoid" Type="Bool">true</Property>
		<Property Name="target.server.tcp.port" Type="Int">3363</Property>
		<Property Name="target.server.tcp.serviceName" Type="Str">Main Application Instance/VI Server</Property>
		<Property Name="target.server.tcp.serviceName.default" Type="Str">Main Application Instance/VI Server</Property>
		<Property Name="target.server.vi.access" Type="Str">+*</Property>
		<Property Name="target.server.vi.callsEnabled" Type="Bool">true</Property>
		<Property Name="target.server.vi.propertiesEnabled" Type="Bool">true</Property>
		<Property Name="target.WebServer.Enabled" Type="Bool">false</Property>
		<Property Name="target.WebServer.LogEnabled" Type="Bool">false</Property>
		<Property Name="target.WebServer.LogPath" Type="Path">/c/ni-rt/system/www/www.log</Property>
		<Property Name="target.WebServer.Port" Type="Int">80</Property>
		<Property Name="target.WebServer.RootPath" Type="Path">/c/ni-rt/system/www</Property>
		<Property Name="target.WebServer.TcpAccess" Type="Str">c+*</Property>
		<Property Name="target.WebServer.Timeout" Type="Int">60</Property>
		<Property Name="target.WebServer.ViAccess" Type="Str">+*</Property>
		<Property Name="target.webservices.SecurityAPIKey" Type="Str">PqVr/ifkAQh+lVrdPIykXlFvg12GhhQFR8H9cUhphgg=:pTe9HRlQuMfJxAG6QCGq7UvoUpJzAzWGKy5SbZ+roSU=</Property>
		<Property Name="target.webservices.ValidTimestampWindow" Type="Int">15</Property>
		<Item Name="Chassis" Type="cRIO Chassis">
			<Property Name="crio.ProgrammingMode" Type="Str">fpga</Property>
			<Property Name="crio.ResourceID" Type="Str">RIO0</Property>
			<Property Name="crio.Type" Type="Str">cRIO-9054</Property>
			<Property Name="NI.SortType" Type="Int">3</Property>
			<Item Name="Real-Time Scan Resources" Type="Module Container">
				<Property Name="crio.ModuleContainerType" Type="Str">crio.RSIModuleContainer</Property>
			</Item>
			<Item Name="Real-Time Resources" Type="Module Container">
				<Property Name="crio.ModuleContainerType" Type="Str">crio.DAQModuleContainer</Property>
			</Item>
			<Item Name="FPGA Target" Type="FPGA Target">
				<Property Name="AutoRun" Type="Bool">false</Property>
				<Property Name="configString.guid" Type="Str">{0108A0FA-EA5D-4E2E-9B5C-738854101D10}"DataType=1000800000000001003C005F03510020000000100001000100000010FFFFFFFFFFFFFFFF0000001F0000000F000000007FFFFFFF00000001FFFFFFF1000000000000000100010000000000000000000000000000;InitDataHash=C72400DF668E56426D1045502381E7C3;Name=Register-B_2;WriteArb=1"{01EDE284-F625-4DAF-A186-F48E839FF44E}Actual Number of Elements=40;ReadArbs=1;WriteArbs=1;Implementation=1;DataType=1000800000000001003C005F03510020000000100001000100000010FFFFFFFFFFFFFFFF0000001F0000000F000000007FFFFFFF00000001FFFFFFF1000000000000000100010000000000000000000000000000;InitDataHash=6230BE6EBA6DB1FA4A9CC9833459AFA7;DRAM Selection=;DRAM Max Outstanding Requests=64;DRAM Include Byte Enables=FALSE;DRAM Grant Time=50;Interface Configuration=Read A-Write B;Multiple Clock Domains=FALSE;Memory Latency=0{028C830A-A095-4346-A0A7-544E131465F0}"ControlLogic=0;NumberOfElements=4101;Type=1;ReadArbs=Arbitrate if Multiple Requestors Only;ElementsPerRead=1;WriteArbs=Arbitrate if Multiple Requestors Only;ElementsPerWrite=1;Implementation=2;FIFO-X;DataType=1000800000000001003C005F03510020000000100001000100000010FFFFFFFFFFFFFFFF0000001F0000000F000000007FFFFFFF00000001FFFFFFF1000000000000000100010000000000000000000000000000;DisableOnOverflowUnderflow=FALSE"{0C5E1042-07A8-4DF0-AD6F-D3EDD24C1389}resource=/crio_Mod1/AI2;0;ReadMethodType=vi.lib\LabVIEW Targets\FPGA\cRIO\shared\nicrio_FXP_Controls\nicrio_FXP_S_24_7.ctl{12B5628B-C73D-4E38-946B-F07CC983EE81}NumberOfSyncRegistersForReadInProject=0;resource=/Time Synchronization/Time Synchronization Fault;0;ReadMethodType=bool{2E53299B-84AB-42B5-9E92-D537425137F4}NumberOfSyncRegistersForReadInProject=0;resource=/Time Synchronization/Time Source;0;ReadMethodType=Targets\NI\FPGA\RIO\CompactRIO\Sync\SyncSource.ctl{3798F8D1-9CEC-49B8-A086-0EF944D35F28}NumberOfSyncRegistersForReadInProject=Auto;resource=/cRIO_Trig/cRIO_Trig6;0;ReadMethodType=bool{3B253AA7-D9D6-45E4-B319-A7EF4E999194}NumberOfSyncRegistersForReadInProject=0;resource=/Time Synchronization/Offset from Time Reference;0;ReadMethodType=i32{44D42E83-4F36-431E-81AA-998E426CB264}Actual Number of Elements=40;ReadArbs=1;WriteArbs=1;Implementation=1;DataType=1000800000000001003C005F03510020000000100001000100000010FFFFFFFFFFFFFFFF0000001F0000000F000000007FFFFFFF00000001FFFFFFF1000000000000000100010000000000000000000000000000;InitDataHash=56484364DDE23929BFC36AE6BF653A38;DRAM Selection=;DRAM Max Outstanding Requests=64;DRAM Include Byte Enables=FALSE;DRAM Grant Time=50;Interface Configuration=Read A-Write B;Multiple Clock Domains=FALSE;Memory Latency=0{45B52DC7-1B6B-458C-BA29-D5BF8548F78B}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/USER FPGA LED;0;ReadMethodType=bool;WriteMethodType=bool{47D8E7DD-4CF9-432C-8810-ABB39830BC57}resource=/crio_Mod1/AI0;0;ReadMethodType=vi.lib\LabVIEW Targets\FPGA\cRIO\shared\nicrio_FXP_Controls\nicrio_FXP_S_24_7.ctl{606B95AA-A63E-4117-8D68-F7F4781E2EE4}NumberOfSyncRegistersForReadInProject=0;resource=/Time Synchronization/Offset from Time Reference Valid;0;ReadMethodType=bool{657EFC45-87EA-4AA9-B7F7-AA10B0EA39FF}NumberOfSyncRegistersForReadInProject=Auto;resource=/12.8 MHz Timebase;0;ReadMethodType=bool{6682DAFE-DE9E-4ED3-BCDB-D114C7198919}NumberOfSyncRegistersForReadInProject=Auto;resource=/10 MHz Timebase;0;ReadMethodType=bool{76FCE484-C688-4869-AB80-F6D1607E8C1F}[crioConfig.Begin]crio.Calibration=1,crio.Location=Slot 1,crio.Type=NI 9229,cRIOModule.ClockSource=1,cRIOModule.DataRate=1,cRIOModule.EnableDECoM=false,cRIOModule.EnableInputFifo=false,cRIOModule.EnableOutputFifo=false,cRIOModule.ExcitationVoltage=1,cRIOModule.ExternalClockSource=&lt;Carrier Clock 12.8 MHz&gt;,cRIOModule.ExtTimeBaseType=,cRIOModule.HalfBridgeEnable=0,cRIOModule.InputConfiguration=0,cRIOModule.RsiAttributes=,cRIOModule.SourceModule=false,cRIOModule.SubPanVisitedVersion=0,cRIOModule.TEDSSupport=false[crioConfig.End]{78EE6008-AB58-42A7-B18E-12F6A98CF225}NumberOfSyncRegistersForReadInProject=0;resource=/Time Synchronization/Time;0;ReadMethodType=u64{8FC19DEA-0D11-4599-9E28-DC65CA8AF064}resource=/System Reset;0;ReadMethodType=bool;WriteMethodType=bool{94F36DD7-2233-4FDE-8092-B11776C65DC4}resource=/Sleep;0;ReadMethodType=bool;WriteMethodType=bool{9553AE3F-A318-4A94-872F-76C42FD27EA6}"ControlLogic=0;NumberOfElements=1023;Type=2;ReadArbs=Arbitrate if Multiple Requestors Only;ElementsPerRead=1;WriteArbs=Arbitrate if Multiple Requestors Only;ElementsPerWrite=1;Implementation=2;FIFO-out;DataType=1000800000000001003C005F03510020000000100001002000000010FFFFFFFF800000000001002000000010000000007FFFFFFF00000001FFFFFFF1000000000000000100010000000000000000000000000000;DisableOnOverflowUnderflow=FALSE"{A903D566-0BD3-454C-95C0-B2EFE1604CC5}resource=/crio_Mod1/Start;0;WriteMethodType=bool{A91C6821-F05C-4670-89E6-A8E5CF47C64F}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/cRIO_Trig/cRIO_Trig3;0;ReadMethodType=bool;WriteMethodType=bool{B39D2A0E-92B9-4AFB-BF77-E37EC8B775D3}resource=/Chassis Temperature;0;ReadMethodType=i16{B657C088-1DAE-48D4-84CE-773611A2C706}resource=/crio_Mod1/AI1;0;ReadMethodType=vi.lib\LabVIEW Targets\FPGA\cRIO\shared\nicrio_FXP_Controls\nicrio_FXP_S_24_7.ctl{B751EED9-7B1B-4B94-B98E-55FE93761ABA}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/cRIO_Trig/cRIO_Trig0;0;ReadMethodType=bool;WriteMethodType=bool{B89C2D4F-C063-4E12-BE2F-32DF9101DF08}resource=/Reset RT App;0;WriteMethodType=bool{BC4AA9AF-18AD-4EFA-99C8-11FC64433606}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/cRIO_Trig/cRIO_Trig2;0;ReadMethodType=bool;WriteMethodType=bool{BDBC90D1-52EE-478C-BBB6-769CA0BA60E8}ResourceName=40 MHz Onboard Clock;TopSignalConnect=Clk40;ClockSignalName=Clk40;MinFreq=40000000.000000;MaxFreq=40000000.000000;VariableFreq=0;NomFreq=40000000.000000;PeakPeriodJitter=250.000000;MinDutyCycle=50.000000;MaxDutyCycle=50.000000;Accuracy=100.000000;RunTime=0;SpreadSpectrum=0;GenericDataHash=D41D8CD98F00B204E9800998ECF8427E;{C4081B4C-8388-44F9-A5D4-ECA53D59D49E}resource=/Scan Clock;0;ReadMethodType=bool{C6F6A12D-A136-4F8C-8F25-AC2358011335}"ControlLogic=0;NumberOfElements=1023;Type=2;ReadArbs=Arbitrate if Multiple Requestors Only;ElementsPerRead=1;WriteArbs=Arbitrate if Multiple Requestors Only;ElementsPerWrite=1;Implementation=2;FIFO address out;DataType=1000800000000001000940030003493332000100000000000000000000;DisableOnOverflowUnderflow=FALSE"{C7DA056B-75EE-429C-809B-E43689F3A198}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/cRIO_Trig/cRIO_Trig1;0;ReadMethodType=bool;WriteMethodType=bool{C93351AA-B198-4983-96FC-1B9642BBF5D5}NumberOfSyncRegistersForReadInProject=Auto;resource=/13.1072 MHz Timebase;0;ReadMethodType=bool{CDAE0FC8-C1CE-4640-B2F8-CEB2A616ED33}"ControlLogic=0;NumberOfElements=8191;Type=2;ReadArbs=Arbitrate if Multiple Requestors Only;ElementsPerRead=1;WriteArbs=Arbitrate if Multiple Requestors Only;ElementsPerWrite=1;Implementation=2;FIFO-y;DataType=1000800000000001003C005F03510021000000110001002100000011FFFFFFFF00000000000100210000001100000000FFFFFFFF00000001FFFFFFF1000000000000000100010000000000000000000000000000;DisableOnOverflowUnderflow=FALSE"{DFB582B4-FA82-42D3-8101-319FEC5DE56C}resource=/crio_Mod1/AI3;0;ReadMethodType=vi.lib\LabVIEW Targets\FPGA\cRIO\shared\nicrio_FXP_Controls\nicrio_FXP_S_24_7.ctl{E20DC649-7E53-4246-9B2D-1FFB9788C051}NumberOfSyncRegistersForReadInProject=Auto;resource=/cRIO_Trig/cRIO_Trig5;0;ReadMethodType=bool{E7E11666-FA67-4229-B696-EA6753A11ADC}NumberOfSyncRegistersForReadInProject=Auto;resource=/cRIO_Trig/cRIO_Trig7;0;ReadMethodType=bool{E86FE605-9A95-4E86-8258-393844E416A7}resource=/crio_Mod1/Stop;0;WriteMethodType=bool{E9C65040-8873-4139-BCD5-30552906293D}NumberOfSyncRegistersForReadInProject=Auto;resource=/cRIO_Trig/cRIO_Trig4;0;ReadMethodType=bool{F71C576A-7CEE-4205-B0EB-CBC2F7B40E67}NumberOfSyncRegistersForReadInProject=Auto;resource=/System Watchdog Expired;0;ReadMethodType=boolcRIO-9054/Clk40/falsefalseFPGA_EXECUTION_MODEFPGA_TARGETFPGA_TARGET_CLASSCRIO_9054FPGA_TARGET_FAMILYARTIX7TARGET_TYPEFPGA</Property>
				<Property Name="configString.name" Type="Str">10 MHz TimebaseNumberOfSyncRegistersForReadInProject=Auto;resource=/10 MHz Timebase;0;ReadMethodType=bool12.8 MHz TimebaseNumberOfSyncRegistersForReadInProject=Auto;resource=/12.8 MHz Timebase;0;ReadMethodType=bool13.1072 MHz TimebaseNumberOfSyncRegistersForReadInProject=Auto;resource=/13.1072 MHz Timebase;0;ReadMethodType=bool40 MHz Onboard ClockResourceName=40 MHz Onboard Clock;TopSignalConnect=Clk40;ClockSignalName=Clk40;MinFreq=40000000.000000;MaxFreq=40000000.000000;VariableFreq=0;NomFreq=40000000.000000;PeakPeriodJitter=250.000000;MinDutyCycle=50.000000;MaxDutyCycle=50.000000;Accuracy=100.000000;RunTime=0;SpreadSpectrum=0;GenericDataHash=D41D8CD98F00B204E9800998ECF8427E;Chassis Temperatureresource=/Chassis Temperature;0;ReadMethodType=i16cRIO_Trig0ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/cRIO_Trig/cRIO_Trig0;0;ReadMethodType=bool;WriteMethodType=boolcRIO_Trig1ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/cRIO_Trig/cRIO_Trig1;0;ReadMethodType=bool;WriteMethodType=boolcRIO_Trig2ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/cRIO_Trig/cRIO_Trig2;0;ReadMethodType=bool;WriteMethodType=boolcRIO_Trig3ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/cRIO_Trig/cRIO_Trig3;0;ReadMethodType=bool;WriteMethodType=boolcRIO_Trig4NumberOfSyncRegistersForReadInProject=Auto;resource=/cRIO_Trig/cRIO_Trig4;0;ReadMethodType=boolcRIO_Trig5NumberOfSyncRegistersForReadInProject=Auto;resource=/cRIO_Trig/cRIO_Trig5;0;ReadMethodType=boolcRIO_Trig6NumberOfSyncRegistersForReadInProject=Auto;resource=/cRIO_Trig/cRIO_Trig6;0;ReadMethodType=boolcRIO_Trig7NumberOfSyncRegistersForReadInProject=Auto;resource=/cRIO_Trig/cRIO_Trig7;0;ReadMethodType=boolcRIO-9054/Clk40/falsefalseFPGA_EXECUTION_MODEFPGA_TARGETFPGA_TARGET_CLASSCRIO_9054FPGA_TARGET_FAMILYARTIX7TARGET_TYPEFPGAFIFO address out"ControlLogic=0;NumberOfElements=1023;Type=2;ReadArbs=Arbitrate if Multiple Requestors Only;ElementsPerRead=1;WriteArbs=Arbitrate if Multiple Requestors Only;ElementsPerWrite=1;Implementation=2;FIFO address out;DataType=1000800000000001000940030003493332000100000000000000000000;DisableOnOverflowUnderflow=FALSE"FIFO-out"ControlLogic=0;NumberOfElements=1023;Type=2;ReadArbs=Arbitrate if Multiple Requestors Only;ElementsPerRead=1;WriteArbs=Arbitrate if Multiple Requestors Only;ElementsPerWrite=1;Implementation=2;FIFO-out;DataType=1000800000000001003C005F03510020000000100001002000000010FFFFFFFF800000000001002000000010000000007FFFFFFF00000001FFFFFFF1000000000000000100010000000000000000000000000000;DisableOnOverflowUnderflow=FALSE"FIFO-X"ControlLogic=0;NumberOfElements=4101;Type=1;ReadArbs=Arbitrate if Multiple Requestors Only;ElementsPerRead=1;WriteArbs=Arbitrate if Multiple Requestors Only;ElementsPerWrite=1;Implementation=2;FIFO-X;DataType=1000800000000001003C005F03510020000000100001000100000010FFFFFFFFFFFFFFFF0000001F0000000F000000007FFFFFFF00000001FFFFFFF1000000000000000100010000000000000000000000000000;DisableOnOverflowUnderflow=FALSE"FIFO-y"ControlLogic=0;NumberOfElements=8191;Type=2;ReadArbs=Arbitrate if Multiple Requestors Only;ElementsPerRead=1;WriteArbs=Arbitrate if Multiple Requestors Only;ElementsPerWrite=1;Implementation=2;FIFO-y;DataType=1000800000000001003C005F03510021000000110001002100000011FFFFFFFF00000000000100210000001100000000FFFFFFFF00000001FFFFFFF1000000000000000100010000000000000000000000000000;DisableOnOverflowUnderflow=FALSE"Memory-B_0Actual Number of Elements=40;ReadArbs=1;WriteArbs=1;Implementation=1;DataType=1000800000000001003C005F03510020000000100001000100000010FFFFFFFFFFFFFFFF0000001F0000000F000000007FFFFFFF00000001FFFFFFF1000000000000000100010000000000000000000000000000;InitDataHash=6230BE6EBA6DB1FA4A9CC9833459AFA7;DRAM Selection=;DRAM Max Outstanding Requests=64;DRAM Include Byte Enables=FALSE;DRAM Grant Time=50;Interface Configuration=Read A-Write B;Multiple Clock Domains=FALSE;Memory Latency=0Memory-B_1Actual Number of Elements=40;ReadArbs=1;WriteArbs=1;Implementation=1;DataType=1000800000000001003C005F03510020000000100001000100000010FFFFFFFFFFFFFFFF0000001F0000000F000000007FFFFFFF00000001FFFFFFF1000000000000000100010000000000000000000000000000;InitDataHash=56484364DDE23929BFC36AE6BF653A38;DRAM Selection=;DRAM Max Outstanding Requests=64;DRAM Include Byte Enables=FALSE;DRAM Grant Time=50;Interface Configuration=Read A-Write B;Multiple Clock Domains=FALSE;Memory Latency=0Mod1/AI0resource=/crio_Mod1/AI0;0;ReadMethodType=vi.lib\LabVIEW Targets\FPGA\cRIO\shared\nicrio_FXP_Controls\nicrio_FXP_S_24_7.ctlMod1/AI1resource=/crio_Mod1/AI1;0;ReadMethodType=vi.lib\LabVIEW Targets\FPGA\cRIO\shared\nicrio_FXP_Controls\nicrio_FXP_S_24_7.ctlMod1/AI2resource=/crio_Mod1/AI2;0;ReadMethodType=vi.lib\LabVIEW Targets\FPGA\cRIO\shared\nicrio_FXP_Controls\nicrio_FXP_S_24_7.ctlMod1/AI3resource=/crio_Mod1/AI3;0;ReadMethodType=vi.lib\LabVIEW Targets\FPGA\cRIO\shared\nicrio_FXP_Controls\nicrio_FXP_S_24_7.ctlMod1/Startresource=/crio_Mod1/Start;0;WriteMethodType=boolMod1/Stopresource=/crio_Mod1/Stop;0;WriteMethodType=boolMod1[crioConfig.Begin]crio.Calibration=1,crio.Location=Slot 1,crio.Type=NI 9229,cRIOModule.ClockSource=1,cRIOModule.DataRate=1,cRIOModule.EnableDECoM=false,cRIOModule.EnableInputFifo=false,cRIOModule.EnableOutputFifo=false,cRIOModule.ExcitationVoltage=1,cRIOModule.ExternalClockSource=&lt;Carrier Clock 12.8 MHz&gt;,cRIOModule.ExtTimeBaseType=,cRIOModule.HalfBridgeEnable=0,cRIOModule.InputConfiguration=0,cRIOModule.RsiAttributes=,cRIOModule.SourceModule=false,cRIOModule.SubPanVisitedVersion=0,cRIOModule.TEDSSupport=false[crioConfig.End]Offset from Time Reference ValidNumberOfSyncRegistersForReadInProject=0;resource=/Time Synchronization/Offset from Time Reference Valid;0;ReadMethodType=boolOffset from Time ReferenceNumberOfSyncRegistersForReadInProject=0;resource=/Time Synchronization/Offset from Time Reference;0;ReadMethodType=i32Register-B_2"DataType=1000800000000001003C005F03510020000000100001000100000010FFFFFFFFFFFFFFFF0000001F0000000F000000007FFFFFFF00000001FFFFFFF1000000000000000100010000000000000000000000000000;InitDataHash=C72400DF668E56426D1045502381E7C3;Name=Register-B_2;WriteArb=1"Reset RT Appresource=/Reset RT App;0;WriteMethodType=boolScan Clockresource=/Scan Clock;0;ReadMethodType=boolSleepresource=/Sleep;0;ReadMethodType=bool;WriteMethodType=boolSystem Resetresource=/System Reset;0;ReadMethodType=bool;WriteMethodType=boolSystem Watchdog ExpiredNumberOfSyncRegistersForReadInProject=Auto;resource=/System Watchdog Expired;0;ReadMethodType=boolTime SourceNumberOfSyncRegistersForReadInProject=0;resource=/Time Synchronization/Time Source;0;ReadMethodType=Targets\NI\FPGA\RIO\CompactRIO\Sync\SyncSource.ctlTime Synchronization FaultNumberOfSyncRegistersForReadInProject=0;resource=/Time Synchronization/Time Synchronization Fault;0;ReadMethodType=boolTimeNumberOfSyncRegistersForReadInProject=0;resource=/Time Synchronization/Time;0;ReadMethodType=u64USER FPGA LEDArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/USER FPGA LED;0;ReadMethodType=bool;WriteMethodType=bool</Property>
				<Property Name="Mode" Type="Int">0</Property>
				<Property Name="NI.LV.FPGA.CompileConfigString" Type="Str">cRIO-9054/Clk40/falsefalseFPGA_EXECUTION_MODEFPGA_TARGETFPGA_TARGET_CLASSCRIO_9054FPGA_TARGET_FAMILYARTIX7TARGET_TYPEFPGA</Property>
				<Property Name="NI.LV.FPGA.Version" Type="Int">6</Property>
				<Property Name="niFpga_TopLevelVIID" Type="Path">/D/github/PART-MLC/MLP/MLP-sizing/v1.2/target.vi</Property>
				<Property Name="Resource Name" Type="Str">RIO0</Property>
				<Property Name="Target Class" Type="Str">cRIO-9054</Property>
				<Property Name="Top-Level Timing Source" Type="Str">40 MHz Onboard Clock</Property>
				<Property Name="Top-Level Timing Source Is Default" Type="Bool">true</Property>
				<Item Name="Chassis I/O" Type="Folder">
					<Item Name="Chassis Temperature" Type="Elemental IO">
						<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="resource">
   <Value>/Chassis Temperature</Value>
   </Attribute>
</AttributeSet>
</Property>
						<Property Name="FPGA.PersistentID" Type="Str">{B39D2A0E-92B9-4AFB-BF77-E37EC8B775D3}</Property>
					</Item>
					<Item Name="Sleep" Type="Elemental IO">
						<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="resource">
   <Value>/Sleep</Value>
   </Attribute>
</AttributeSet>
</Property>
						<Property Name="FPGA.PersistentID" Type="Str">{94F36DD7-2233-4FDE-8092-B11776C65DC4}</Property>
					</Item>
					<Item Name="System Reset" Type="Elemental IO">
						<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="resource">
   <Value>/System Reset</Value>
   </Attribute>
</AttributeSet>
</Property>
						<Property Name="FPGA.PersistentID" Type="Str">{8FC19DEA-0D11-4599-9E28-DC65CA8AF064}</Property>
					</Item>
					<Item Name="Scan Clock" Type="Elemental IO">
						<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="resource">
   <Value>/Scan Clock</Value>
   </Attribute>
</AttributeSet>
</Property>
						<Property Name="FPGA.PersistentID" Type="Str">{C4081B4C-8388-44F9-A5D4-ECA53D59D49E}</Property>
					</Item>
					<Item Name="Reset RT App" Type="Elemental IO">
						<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="resource">
   <Value>/Reset RT App</Value>
   </Attribute>
</AttributeSet>
</Property>
						<Property Name="FPGA.PersistentID" Type="Str">{B89C2D4F-C063-4E12-BE2F-32DF9101DF08}</Property>
					</Item>
					<Item Name="System Watchdog Expired" Type="Elemental IO">
						<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="NumberOfSyncRegistersForReadInProject">
   <Value>Auto</Value>
   </Attribute>
   <Attribute name="resource">
   <Value>/System Watchdog Expired</Value>
   </Attribute>
</AttributeSet>
</Property>
						<Property Name="FPGA.PersistentID" Type="Str">{F71C576A-7CEE-4205-B0EB-CBC2F7B40E67}</Property>
					</Item>
					<Item Name="12.8 MHz Timebase" Type="Elemental IO">
						<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="NumberOfSyncRegistersForReadInProject">
   <Value>Auto</Value>
   </Attribute>
   <Attribute name="resource">
   <Value>/12.8 MHz Timebase</Value>
   </Attribute>
</AttributeSet>
</Property>
						<Property Name="FPGA.PersistentID" Type="Str">{657EFC45-87EA-4AA9-B7F7-AA10B0EA39FF}</Property>
					</Item>
					<Item Name="13.1072 MHz Timebase" Type="Elemental IO">
						<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="NumberOfSyncRegistersForReadInProject">
   <Value>Auto</Value>
   </Attribute>
   <Attribute name="resource">
   <Value>/13.1072 MHz Timebase</Value>
   </Attribute>
</AttributeSet>
</Property>
						<Property Name="FPGA.PersistentID" Type="Str">{C93351AA-B198-4983-96FC-1B9642BBF5D5}</Property>
					</Item>
					<Item Name="10 MHz Timebase" Type="Elemental IO">
						<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="NumberOfSyncRegistersForReadInProject">
   <Value>Auto</Value>
   </Attribute>
   <Attribute name="resource">
   <Value>/10 MHz Timebase</Value>
   </Attribute>
</AttributeSet>
</Property>
						<Property Name="FPGA.PersistentID" Type="Str">{6682DAFE-DE9E-4ED3-BCDB-D114C7198919}</Property>
					</Item>
					<Item Name="USER FPGA LED" Type="Elemental IO">
						<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="ArbitrationForOutputData">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputData">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForReadInProject">
   <Value>Auto</Value>
   </Attribute>
   <Attribute name="resource">
   <Value>/USER FPGA LED</Value>
   </Attribute>
</AttributeSet>
</Property>
						<Property Name="FPGA.PersistentID" Type="Str">{45B52DC7-1B6B-458C-BA29-D5BF8548F78B}</Property>
					</Item>
				</Item>
				<Item Name="cRIO_Trig" Type="Folder">
					<Item Name="cRIO_Trig0" Type="Elemental IO">
						<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="ArbitrationForOutputData">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputData">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForReadInProject">
   <Value>Auto</Value>
   </Attribute>
   <Attribute name="resource">
   <Value>/cRIO_Trig/cRIO_Trig0</Value>
   </Attribute>
</AttributeSet>
</Property>
						<Property Name="FPGA.PersistentID" Type="Str">{B751EED9-7B1B-4B94-B98E-55FE93761ABA}</Property>
					</Item>
					<Item Name="cRIO_Trig1" Type="Elemental IO">
						<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="ArbitrationForOutputData">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputData">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForReadInProject">
   <Value>Auto</Value>
   </Attribute>
   <Attribute name="resource">
   <Value>/cRIO_Trig/cRIO_Trig1</Value>
   </Attribute>
</AttributeSet>
</Property>
						<Property Name="FPGA.PersistentID" Type="Str">{C7DA056B-75EE-429C-809B-E43689F3A198}</Property>
					</Item>
					<Item Name="cRIO_Trig2" Type="Elemental IO">
						<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="ArbitrationForOutputData">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputData">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForReadInProject">
   <Value>Auto</Value>
   </Attribute>
   <Attribute name="resource">
   <Value>/cRIO_Trig/cRIO_Trig2</Value>
   </Attribute>
</AttributeSet>
</Property>
						<Property Name="FPGA.PersistentID" Type="Str">{BC4AA9AF-18AD-4EFA-99C8-11FC64433606}</Property>
					</Item>
					<Item Name="cRIO_Trig3" Type="Elemental IO">
						<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="ArbitrationForOutputData">
   <Value>NeverArbitrate</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForOutputData">
   <Value>1</Value>
   </Attribute>
   <Attribute name="NumberOfSyncRegistersForReadInProject">
   <Value>Auto</Value>
   </Attribute>
   <Attribute name="resource">
   <Value>/cRIO_Trig/cRIO_Trig3</Value>
   </Attribute>
</AttributeSet>
</Property>
						<Property Name="FPGA.PersistentID" Type="Str">{A91C6821-F05C-4670-89E6-A8E5CF47C64F}</Property>
					</Item>
					<Item Name="cRIO_Trig4" Type="Elemental IO">
						<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="NumberOfSyncRegistersForReadInProject">
   <Value>Auto</Value>
   </Attribute>
   <Attribute name="resource">
   <Value>/cRIO_Trig/cRIO_Trig4</Value>
   </Attribute>
</AttributeSet>
</Property>
						<Property Name="FPGA.PersistentID" Type="Str">{E9C65040-8873-4139-BCD5-30552906293D}</Property>
					</Item>
					<Item Name="cRIO_Trig5" Type="Elemental IO">
						<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="NumberOfSyncRegistersForReadInProject">
   <Value>Auto</Value>
   </Attribute>
   <Attribute name="resource">
   <Value>/cRIO_Trig/cRIO_Trig5</Value>
   </Attribute>
</AttributeSet>
</Property>
						<Property Name="FPGA.PersistentID" Type="Str">{E20DC649-7E53-4246-9B2D-1FFB9788C051}</Property>
					</Item>
					<Item Name="cRIO_Trig6" Type="Elemental IO">
						<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="NumberOfSyncRegistersForReadInProject">
   <Value>Auto</Value>
   </Attribute>
   <Attribute name="resource">
   <Value>/cRIO_Trig/cRIO_Trig6</Value>
   </Attribute>
</AttributeSet>
</Property>
						<Property Name="FPGA.PersistentID" Type="Str">{3798F8D1-9CEC-49B8-A086-0EF944D35F28}</Property>
					</Item>
					<Item Name="cRIO_Trig7" Type="Elemental IO">
						<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="NumberOfSyncRegistersForReadInProject">
   <Value>Auto</Value>
   </Attribute>
   <Attribute name="resource">
   <Value>/cRIO_Trig/cRIO_Trig7</Value>
   </Attribute>
</AttributeSet>
</Property>
						<Property Name="FPGA.PersistentID" Type="Str">{E7E11666-FA67-4229-B696-EA6753A11ADC}</Property>
					</Item>
				</Item>
				<Item Name="Time Synchronization" Type="Folder">
					<Item Name="Time" Type="Elemental IO">
						<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="NumberOfSyncRegistersForReadInProject">
   <Value>0</Value>
   </Attribute>
   <Attribute name="resource">
   <Value>/Time Synchronization/Time</Value>
   </Attribute>
</AttributeSet>
</Property>
						<Property Name="FPGA.PersistentID" Type="Str">{78EE6008-AB58-42A7-B18E-12F6A98CF225}</Property>
					</Item>
					<Item Name="Time Source" Type="Elemental IO">
						<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="NumberOfSyncRegistersForReadInProject">
   <Value>0</Value>
   </Attribute>
   <Attribute name="resource">
   <Value>/Time Synchronization/Time Source</Value>
   </Attribute>
</AttributeSet>
</Property>
						<Property Name="FPGA.PersistentID" Type="Str">{2E53299B-84AB-42B5-9E92-D537425137F4}</Property>
					</Item>
					<Item Name="Time Synchronization Fault" Type="Elemental IO">
						<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="NumberOfSyncRegistersForReadInProject">
   <Value>0</Value>
   </Attribute>
   <Attribute name="resource">
   <Value>/Time Synchronization/Time Synchronization Fault</Value>
   </Attribute>
</AttributeSet>
</Property>
						<Property Name="FPGA.PersistentID" Type="Str">{12B5628B-C73D-4E38-946B-F07CC983EE81}</Property>
					</Item>
					<Item Name="Offset from Time Reference" Type="Elemental IO">
						<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="NumberOfSyncRegistersForReadInProject">
   <Value>0</Value>
   </Attribute>
   <Attribute name="resource">
   <Value>/Time Synchronization/Offset from Time Reference</Value>
   </Attribute>
</AttributeSet>
</Property>
						<Property Name="FPGA.PersistentID" Type="Str">{3B253AA7-D9D6-45E4-B319-A7EF4E999194}</Property>
					</Item>
					<Item Name="Offset from Time Reference Valid" Type="Elemental IO">
						<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="NumberOfSyncRegistersForReadInProject">
   <Value>0</Value>
   </Attribute>
   <Attribute name="resource">
   <Value>/Time Synchronization/Offset from Time Reference Valid</Value>
   </Attribute>
</AttributeSet>
</Property>
						<Property Name="FPGA.PersistentID" Type="Str">{606B95AA-A63E-4117-8D68-F7F4781E2EE4}</Property>
					</Item>
				</Item>
				<Item Name="Mod1" Type="Folder">
					<Item Name="Mod1/AI0" Type="Elemental IO">
						<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="resource">
   <Value>/crio_Mod1/AI0</Value>
   </Attribute>
</AttributeSet>
</Property>
						<Property Name="FPGA.PersistentID" Type="Str">{47D8E7DD-4CF9-432C-8810-ABB39830BC57}</Property>
					</Item>
					<Item Name="Mod1/AI1" Type="Elemental IO">
						<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="resource">
   <Value>/crio_Mod1/AI1</Value>
   </Attribute>
</AttributeSet>
</Property>
						<Property Name="FPGA.PersistentID" Type="Str">{B657C088-1DAE-48D4-84CE-773611A2C706}</Property>
					</Item>
					<Item Name="Mod1/AI2" Type="Elemental IO">
						<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="resource">
   <Value>/crio_Mod1/AI2</Value>
   </Attribute>
</AttributeSet>
</Property>
						<Property Name="FPGA.PersistentID" Type="Str">{0C5E1042-07A8-4DF0-AD6F-D3EDD24C1389}</Property>
					</Item>
					<Item Name="Mod1/AI3" Type="Elemental IO">
						<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="resource">
   <Value>/crio_Mod1/AI3</Value>
   </Attribute>
</AttributeSet>
</Property>
						<Property Name="FPGA.PersistentID" Type="Str">{DFB582B4-FA82-42D3-8101-319FEC5DE56C}</Property>
					</Item>
					<Item Name="Mod1/Start" Type="Elemental IO">
						<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="resource">
   <Value>/crio_Mod1/Start</Value>
   </Attribute>
</AttributeSet>
</Property>
						<Property Name="FPGA.PersistentID" Type="Str">{A903D566-0BD3-454C-95C0-B2EFE1604CC5}</Property>
					</Item>
					<Item Name="Mod1/Stop" Type="Elemental IO">
						<Property Name="eioAttrBag" Type="Xml"><AttributeSet name="">
   <Attribute name="resource">
   <Value>/crio_Mod1/Stop</Value>
   </Attribute>
</AttributeSet>
</Property>
						<Property Name="FPGA.PersistentID" Type="Str">{E86FE605-9A95-4E86-8258-393844E416A7}</Property>
					</Item>
				</Item>
				<Item Name="40 MHz Onboard Clock" Type="FPGA Base Clock">
					<Property Name="FPGA.PersistentID" Type="Str">{BDBC90D1-52EE-478C-BBB6-769CA0BA60E8}</Property>
					<Property Name="NI.LV.FPGA.BaseTSConfig" Type="Str">ResourceName=40 MHz Onboard Clock;TopSignalConnect=Clk40;ClockSignalName=Clk40;MinFreq=40000000.000000;MaxFreq=40000000.000000;VariableFreq=0;NomFreq=40000000.000000;PeakPeriodJitter=250.000000;MinDutyCycle=50.000000;MaxDutyCycle=50.000000;Accuracy=100.000000;RunTime=0;SpreadSpectrum=0;GenericDataHash=D41D8CD98F00B204E9800998ECF8427E;</Property>
					<Property Name="NI.LV.FPGA.BaseTSConfig.Accuracy" Type="Dbl">100</Property>
					<Property Name="NI.LV.FPGA.BaseTSConfig.ClockSignalName" Type="Str">Clk40</Property>
					<Property Name="NI.LV.FPGA.BaseTSConfig.MaxDutyCycle" Type="Dbl">50</Property>
					<Property Name="NI.LV.FPGA.BaseTSConfig.MaxFrequency" Type="Dbl">40000000</Property>
					<Property Name="NI.LV.FPGA.BaseTSConfig.MinDutyCycle" Type="Dbl">50</Property>
					<Property Name="NI.LV.FPGA.BaseTSConfig.MinFrequency" Type="Dbl">40000000</Property>
					<Property Name="NI.LV.FPGA.BaseTSConfig.NominalFrequency" Type="Dbl">40000000</Property>
					<Property Name="NI.LV.FPGA.BaseTSConfig.PeakPeriodJitter" Type="Dbl">250</Property>
					<Property Name="NI.LV.FPGA.BaseTSConfig.ResourceName" Type="Str">40 MHz Onboard Clock</Property>
					<Property Name="NI.LV.FPGA.BaseTSConfig.SupportAndRequireRuntimeEnableDisable" Type="Bool">false</Property>
					<Property Name="NI.LV.FPGA.BaseTSConfig.TopSignalConnect" Type="Str">Clk40</Property>
					<Property Name="NI.LV.FPGA.BaseTSConfig.VariableFrequency" Type="Bool">false</Property>
					<Property Name="NI.LV.FPGA.Valid" Type="Bool">true</Property>
					<Property Name="NI.LV.FPGA.Version" Type="Int">5</Property>
				</Item>
				<Item Name="Mod1" Type="RIO C Series Module">
					<Property Name="crio.Calibration" Type="Str">1</Property>
					<Property Name="crio.Location" Type="Str">Slot 1</Property>
					<Property Name="crio.RequiresValidation" Type="Bool">true</Property>
					<Property Name="crio.SDcounterSlaveChannelMask" Type="Str">0</Property>
					<Property Name="crio.SDCounterSlaveMasterSlot" Type="Str">0</Property>
					<Property Name="crio.SDInputFilter" Type="Str">128</Property>
					<Property Name="crio.SupportsDynamicRes" Type="Bool">true</Property>
					<Property Name="crio.Type" Type="Str">NI 9229</Property>
					<Property Name="cRIOModule.ClockSource" Type="Str">1</Property>
					<Property Name="cRIOModule.DataRate" Type="Str">1</Property>
					<Property Name="cRIOModule.DigitalIOMode" Type="Str">0</Property>
					<Property Name="cRIOModule.EnableSpecialtyDigital" Type="Str">false</Property>
					<Property Name="cRIOModule.ExcitationVoltage" Type="Str">1</Property>
					<Property Name="cRIOModule.ExternalClockSource" Type="Str">&lt;Carrier Clock 12.8 MHz&gt;</Property>
					<Property Name="cRIOModule.ExtTimeBaseType" Type="Str"></Property>
					<Property Name="cRIOModule.HalfBridgeEnable" Type="Str">0</Property>
					<Property Name="cRIOModule.InputConfiguration" Type="Str">0</Property>
					<Property Name="cRIOModule.SourceModule" Type="Str">false</Property>
					<Property Name="cRIOModule.SubPanVisitedVersion" Type="Str">0</Property>
					<Property Name="cRIOModule.TEDSSupport" Type="Str">false</Property>
					<Property Name="FPGA.PersistentID" Type="Str">{76FCE484-C688-4869-AB80-F6D1607E8C1F}</Property>
				</Item>
				<Item Name="target.vi" Type="VI" URL="../target.vi">
					<Property Name="configString.guid" Type="Str">{0108A0FA-EA5D-4E2E-9B5C-738854101D10}"DataType=1000800000000001003C005F03510020000000100001000100000010FFFFFFFFFFFFFFFF0000001F0000000F000000007FFFFFFF00000001FFFFFFF1000000000000000100010000000000000000000000000000;InitDataHash=C72400DF668E56426D1045502381E7C3;Name=Register-B_2;WriteArb=1"{01EDE284-F625-4DAF-A186-F48E839FF44E}Actual Number of Elements=40;ReadArbs=1;WriteArbs=1;Implementation=1;DataType=1000800000000001003C005F03510020000000100001000100000010FFFFFFFFFFFFFFFF0000001F0000000F000000007FFFFFFF00000001FFFFFFF1000000000000000100010000000000000000000000000000;InitDataHash=6230BE6EBA6DB1FA4A9CC9833459AFA7;DRAM Selection=;DRAM Max Outstanding Requests=64;DRAM Include Byte Enables=FALSE;DRAM Grant Time=50;Interface Configuration=Read A-Write B;Multiple Clock Domains=FALSE;Memory Latency=0{028C830A-A095-4346-A0A7-544E131465F0}"ControlLogic=0;NumberOfElements=4101;Type=1;ReadArbs=Arbitrate if Multiple Requestors Only;ElementsPerRead=1;WriteArbs=Arbitrate if Multiple Requestors Only;ElementsPerWrite=1;Implementation=2;FIFO-X;DataType=1000800000000001003C005F03510020000000100001000100000010FFFFFFFFFFFFFFFF0000001F0000000F000000007FFFFFFF00000001FFFFFFF1000000000000000100010000000000000000000000000000;DisableOnOverflowUnderflow=FALSE"{0C5E1042-07A8-4DF0-AD6F-D3EDD24C1389}resource=/crio_Mod1/AI2;0;ReadMethodType=vi.lib\LabVIEW Targets\FPGA\cRIO\shared\nicrio_FXP_Controls\nicrio_FXP_S_24_7.ctl{12B5628B-C73D-4E38-946B-F07CC983EE81}NumberOfSyncRegistersForReadInProject=0;resource=/Time Synchronization/Time Synchronization Fault;0;ReadMethodType=bool{2E53299B-84AB-42B5-9E92-D537425137F4}NumberOfSyncRegistersForReadInProject=0;resource=/Time Synchronization/Time Source;0;ReadMethodType=Targets\NI\FPGA\RIO\CompactRIO\Sync\SyncSource.ctl{3798F8D1-9CEC-49B8-A086-0EF944D35F28}NumberOfSyncRegistersForReadInProject=Auto;resource=/cRIO_Trig/cRIO_Trig6;0;ReadMethodType=bool{3B253AA7-D9D6-45E4-B319-A7EF4E999194}NumberOfSyncRegistersForReadInProject=0;resource=/Time Synchronization/Offset from Time Reference;0;ReadMethodType=i32{44D42E83-4F36-431E-81AA-998E426CB264}Actual Number of Elements=40;ReadArbs=1;WriteArbs=1;Implementation=1;DataType=1000800000000001003C005F03510020000000100001000100000010FFFFFFFFFFFFFFFF0000001F0000000F000000007FFFFFFF00000001FFFFFFF1000000000000000100010000000000000000000000000000;InitDataHash=56484364DDE23929BFC36AE6BF653A38;DRAM Selection=;DRAM Max Outstanding Requests=64;DRAM Include Byte Enables=FALSE;DRAM Grant Time=50;Interface Configuration=Read A-Write B;Multiple Clock Domains=FALSE;Memory Latency=0{45B52DC7-1B6B-458C-BA29-D5BF8548F78B}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/USER FPGA LED;0;ReadMethodType=bool;WriteMethodType=bool{47D8E7DD-4CF9-432C-8810-ABB39830BC57}resource=/crio_Mod1/AI0;0;ReadMethodType=vi.lib\LabVIEW Targets\FPGA\cRIO\shared\nicrio_FXP_Controls\nicrio_FXP_S_24_7.ctl{606B95AA-A63E-4117-8D68-F7F4781E2EE4}NumberOfSyncRegistersForReadInProject=0;resource=/Time Synchronization/Offset from Time Reference Valid;0;ReadMethodType=bool{657EFC45-87EA-4AA9-B7F7-AA10B0EA39FF}NumberOfSyncRegistersForReadInProject=Auto;resource=/12.8 MHz Timebase;0;ReadMethodType=bool{6682DAFE-DE9E-4ED3-BCDB-D114C7198919}NumberOfSyncRegistersForReadInProject=Auto;resource=/10 MHz Timebase;0;ReadMethodType=bool{76FCE484-C688-4869-AB80-F6D1607E8C1F}[crioConfig.Begin]crio.Calibration=1,crio.Location=Slot 1,crio.Type=NI 9229,cRIOModule.ClockSource=1,cRIOModule.DataRate=1,cRIOModule.EnableDECoM=false,cRIOModule.EnableInputFifo=false,cRIOModule.EnableOutputFifo=false,cRIOModule.ExcitationVoltage=1,cRIOModule.ExternalClockSource=&lt;Carrier Clock 12.8 MHz&gt;,cRIOModule.ExtTimeBaseType=,cRIOModule.HalfBridgeEnable=0,cRIOModule.InputConfiguration=0,cRIOModule.RsiAttributes=,cRIOModule.SourceModule=false,cRIOModule.SubPanVisitedVersion=0,cRIOModule.TEDSSupport=false[crioConfig.End]{78EE6008-AB58-42A7-B18E-12F6A98CF225}NumberOfSyncRegistersForReadInProject=0;resource=/Time Synchronization/Time;0;ReadMethodType=u64{8FC19DEA-0D11-4599-9E28-DC65CA8AF064}resource=/System Reset;0;ReadMethodType=bool;WriteMethodType=bool{94F36DD7-2233-4FDE-8092-B11776C65DC4}resource=/Sleep;0;ReadMethodType=bool;WriteMethodType=bool{9553AE3F-A318-4A94-872F-76C42FD27EA6}"ControlLogic=0;NumberOfElements=1023;Type=2;ReadArbs=Arbitrate if Multiple Requestors Only;ElementsPerRead=1;WriteArbs=Arbitrate if Multiple Requestors Only;ElementsPerWrite=1;Implementation=2;FIFO-out;DataType=1000800000000001003C005F03510020000000100001002000000010FFFFFFFF800000000001002000000010000000007FFFFFFF00000001FFFFFFF1000000000000000100010000000000000000000000000000;DisableOnOverflowUnderflow=FALSE"{A903D566-0BD3-454C-95C0-B2EFE1604CC5}resource=/crio_Mod1/Start;0;WriteMethodType=bool{A91C6821-F05C-4670-89E6-A8E5CF47C64F}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/cRIO_Trig/cRIO_Trig3;0;ReadMethodType=bool;WriteMethodType=bool{B39D2A0E-92B9-4AFB-BF77-E37EC8B775D3}resource=/Chassis Temperature;0;ReadMethodType=i16{B657C088-1DAE-48D4-84CE-773611A2C706}resource=/crio_Mod1/AI1;0;ReadMethodType=vi.lib\LabVIEW Targets\FPGA\cRIO\shared\nicrio_FXP_Controls\nicrio_FXP_S_24_7.ctl{B751EED9-7B1B-4B94-B98E-55FE93761ABA}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/cRIO_Trig/cRIO_Trig0;0;ReadMethodType=bool;WriteMethodType=bool{B89C2D4F-C063-4E12-BE2F-32DF9101DF08}resource=/Reset RT App;0;WriteMethodType=bool{BC4AA9AF-18AD-4EFA-99C8-11FC64433606}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/cRIO_Trig/cRIO_Trig2;0;ReadMethodType=bool;WriteMethodType=bool{BDBC90D1-52EE-478C-BBB6-769CA0BA60E8}ResourceName=40 MHz Onboard Clock;TopSignalConnect=Clk40;ClockSignalName=Clk40;MinFreq=40000000.000000;MaxFreq=40000000.000000;VariableFreq=0;NomFreq=40000000.000000;PeakPeriodJitter=250.000000;MinDutyCycle=50.000000;MaxDutyCycle=50.000000;Accuracy=100.000000;RunTime=0;SpreadSpectrum=0;GenericDataHash=D41D8CD98F00B204E9800998ECF8427E;{C4081B4C-8388-44F9-A5D4-ECA53D59D49E}resource=/Scan Clock;0;ReadMethodType=bool{C6F6A12D-A136-4F8C-8F25-AC2358011335}"ControlLogic=0;NumberOfElements=1023;Type=2;ReadArbs=Arbitrate if Multiple Requestors Only;ElementsPerRead=1;WriteArbs=Arbitrate if Multiple Requestors Only;ElementsPerWrite=1;Implementation=2;FIFO address out;DataType=1000800000000001000940030003493332000100000000000000000000;DisableOnOverflowUnderflow=FALSE"{C7DA056B-75EE-429C-809B-E43689F3A198}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/cRIO_Trig/cRIO_Trig1;0;ReadMethodType=bool;WriteMethodType=bool{C93351AA-B198-4983-96FC-1B9642BBF5D5}NumberOfSyncRegistersForReadInProject=Auto;resource=/13.1072 MHz Timebase;0;ReadMethodType=bool{CDAE0FC8-C1CE-4640-B2F8-CEB2A616ED33}"ControlLogic=0;NumberOfElements=8191;Type=2;ReadArbs=Arbitrate if Multiple Requestors Only;ElementsPerRead=1;WriteArbs=Arbitrate if Multiple Requestors Only;ElementsPerWrite=1;Implementation=2;FIFO-y;DataType=1000800000000001003C005F03510021000000110001002100000011FFFFFFFF00000000000100210000001100000000FFFFFFFF00000001FFFFFFF1000000000000000100010000000000000000000000000000;DisableOnOverflowUnderflow=FALSE"{DFB582B4-FA82-42D3-8101-319FEC5DE56C}resource=/crio_Mod1/AI3;0;ReadMethodType=vi.lib\LabVIEW Targets\FPGA\cRIO\shared\nicrio_FXP_Controls\nicrio_FXP_S_24_7.ctl{E20DC649-7E53-4246-9B2D-1FFB9788C051}NumberOfSyncRegistersForReadInProject=Auto;resource=/cRIO_Trig/cRIO_Trig5;0;ReadMethodType=bool{E7E11666-FA67-4229-B696-EA6753A11ADC}NumberOfSyncRegistersForReadInProject=Auto;resource=/cRIO_Trig/cRIO_Trig7;0;ReadMethodType=bool{E86FE605-9A95-4E86-8258-393844E416A7}resource=/crio_Mod1/Stop;0;WriteMethodType=bool{E9C65040-8873-4139-BCD5-30552906293D}NumberOfSyncRegistersForReadInProject=Auto;resource=/cRIO_Trig/cRIO_Trig4;0;ReadMethodType=bool{F71C576A-7CEE-4205-B0EB-CBC2F7B40E67}NumberOfSyncRegistersForReadInProject=Auto;resource=/System Watchdog Expired;0;ReadMethodType=boolcRIO-9054/Clk40/falsefalseFPGA_EXECUTION_MODEFPGA_TARGETFPGA_TARGET_CLASSCRIO_9054FPGA_TARGET_FAMILYARTIX7TARGET_TYPEFPGA</Property>
					<Property Name="configString.name" Type="Str">10 MHz TimebaseNumberOfSyncRegistersForReadInProject=Auto;resource=/10 MHz Timebase;0;ReadMethodType=bool12.8 MHz TimebaseNumberOfSyncRegistersForReadInProject=Auto;resource=/12.8 MHz Timebase;0;ReadMethodType=bool13.1072 MHz TimebaseNumberOfSyncRegistersForReadInProject=Auto;resource=/13.1072 MHz Timebase;0;ReadMethodType=bool40 MHz Onboard ClockResourceName=40 MHz Onboard Clock;TopSignalConnect=Clk40;ClockSignalName=Clk40;MinFreq=40000000.000000;MaxFreq=40000000.000000;VariableFreq=0;NomFreq=40000000.000000;PeakPeriodJitter=250.000000;MinDutyCycle=50.000000;MaxDutyCycle=50.000000;Accuracy=100.000000;RunTime=0;SpreadSpectrum=0;GenericDataHash=D41D8CD98F00B204E9800998ECF8427E;Chassis Temperatureresource=/Chassis Temperature;0;ReadMethodType=i16cRIO_Trig0ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/cRIO_Trig/cRIO_Trig0;0;ReadMethodType=bool;WriteMethodType=boolcRIO_Trig1ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/cRIO_Trig/cRIO_Trig1;0;ReadMethodType=bool;WriteMethodType=boolcRIO_Trig2ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/cRIO_Trig/cRIO_Trig2;0;ReadMethodType=bool;WriteMethodType=boolcRIO_Trig3ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/cRIO_Trig/cRIO_Trig3;0;ReadMethodType=bool;WriteMethodType=boolcRIO_Trig4NumberOfSyncRegistersForReadInProject=Auto;resource=/cRIO_Trig/cRIO_Trig4;0;ReadMethodType=boolcRIO_Trig5NumberOfSyncRegistersForReadInProject=Auto;resource=/cRIO_Trig/cRIO_Trig5;0;ReadMethodType=boolcRIO_Trig6NumberOfSyncRegistersForReadInProject=Auto;resource=/cRIO_Trig/cRIO_Trig6;0;ReadMethodType=boolcRIO_Trig7NumberOfSyncRegistersForReadInProject=Auto;resource=/cRIO_Trig/cRIO_Trig7;0;ReadMethodType=boolcRIO-9054/Clk40/falsefalseFPGA_EXECUTION_MODEFPGA_TARGETFPGA_TARGET_CLASSCRIO_9054FPGA_TARGET_FAMILYARTIX7TARGET_TYPEFPGAFIFO address out"ControlLogic=0;NumberOfElements=1023;Type=2;ReadArbs=Arbitrate if Multiple Requestors Only;ElementsPerRead=1;WriteArbs=Arbitrate if Multiple Requestors Only;ElementsPerWrite=1;Implementation=2;FIFO address out;DataType=1000800000000001000940030003493332000100000000000000000000;DisableOnOverflowUnderflow=FALSE"FIFO-out"ControlLogic=0;NumberOfElements=1023;Type=2;ReadArbs=Arbitrate if Multiple Requestors Only;ElementsPerRead=1;WriteArbs=Arbitrate if Multiple Requestors Only;ElementsPerWrite=1;Implementation=2;FIFO-out;DataType=1000800000000001003C005F03510020000000100001002000000010FFFFFFFF800000000001002000000010000000007FFFFFFF00000001FFFFFFF1000000000000000100010000000000000000000000000000;DisableOnOverflowUnderflow=FALSE"FIFO-X"ControlLogic=0;NumberOfElements=4101;Type=1;ReadArbs=Arbitrate if Multiple Requestors Only;ElementsPerRead=1;WriteArbs=Arbitrate if Multiple Requestors Only;ElementsPerWrite=1;Implementation=2;FIFO-X;DataType=1000800000000001003C005F03510020000000100001000100000010FFFFFFFFFFFFFFFF0000001F0000000F000000007FFFFFFF00000001FFFFFFF1000000000000000100010000000000000000000000000000;DisableOnOverflowUnderflow=FALSE"FIFO-y"ControlLogic=0;NumberOfElements=8191;Type=2;ReadArbs=Arbitrate if Multiple Requestors Only;ElementsPerRead=1;WriteArbs=Arbitrate if Multiple Requestors Only;ElementsPerWrite=1;Implementation=2;FIFO-y;DataType=1000800000000001003C005F03510021000000110001002100000011FFFFFFFF00000000000100210000001100000000FFFFFFFF00000001FFFFFFF1000000000000000100010000000000000000000000000000;DisableOnOverflowUnderflow=FALSE"Memory-B_0Actual Number of Elements=40;ReadArbs=1;WriteArbs=1;Implementation=1;DataType=1000800000000001003C005F03510020000000100001000100000010FFFFFFFFFFFFFFFF0000001F0000000F000000007FFFFFFF00000001FFFFFFF1000000000000000100010000000000000000000000000000;InitDataHash=6230BE6EBA6DB1FA4A9CC9833459AFA7;DRAM Selection=;DRAM Max Outstanding Requests=64;DRAM Include Byte Enables=FALSE;DRAM Grant Time=50;Interface Configuration=Read A-Write B;Multiple Clock Domains=FALSE;Memory Latency=0Memory-B_1Actual Number of Elements=40;ReadArbs=1;WriteArbs=1;Implementation=1;DataType=1000800000000001003C005F03510020000000100001000100000010FFFFFFFFFFFFFFFF0000001F0000000F000000007FFFFFFF00000001FFFFFFF1000000000000000100010000000000000000000000000000;InitDataHash=56484364DDE23929BFC36AE6BF653A38;DRAM Selection=;DRAM Max Outstanding Requests=64;DRAM Include Byte Enables=FALSE;DRAM Grant Time=50;Interface Configuration=Read A-Write B;Multiple Clock Domains=FALSE;Memory Latency=0Mod1/AI0resource=/crio_Mod1/AI0;0;ReadMethodType=vi.lib\LabVIEW Targets\FPGA\cRIO\shared\nicrio_FXP_Controls\nicrio_FXP_S_24_7.ctlMod1/AI1resource=/crio_Mod1/AI1;0;ReadMethodType=vi.lib\LabVIEW Targets\FPGA\cRIO\shared\nicrio_FXP_Controls\nicrio_FXP_S_24_7.ctlMod1/AI2resource=/crio_Mod1/AI2;0;ReadMethodType=vi.lib\LabVIEW Targets\FPGA\cRIO\shared\nicrio_FXP_Controls\nicrio_FXP_S_24_7.ctlMod1/AI3resource=/crio_Mod1/AI3;0;ReadMethodType=vi.lib\LabVIEW Targets\FPGA\cRIO\shared\nicrio_FXP_Controls\nicrio_FXP_S_24_7.ctlMod1/Startresource=/crio_Mod1/Start;0;WriteMethodType=boolMod1/Stopresource=/crio_Mod1/Stop;0;WriteMethodType=boolMod1[crioConfig.Begin]crio.Calibration=1,crio.Location=Slot 1,crio.Type=NI 9229,cRIOModule.ClockSource=1,cRIOModule.DataRate=1,cRIOModule.EnableDECoM=false,cRIOModule.EnableInputFifo=false,cRIOModule.EnableOutputFifo=false,cRIOModule.ExcitationVoltage=1,cRIOModule.ExternalClockSource=&lt;Carrier Clock 12.8 MHz&gt;,cRIOModule.ExtTimeBaseType=,cRIOModule.HalfBridgeEnable=0,cRIOModule.InputConfiguration=0,cRIOModule.RsiAttributes=,cRIOModule.SourceModule=false,cRIOModule.SubPanVisitedVersion=0,cRIOModule.TEDSSupport=false[crioConfig.End]Offset from Time Reference ValidNumberOfSyncRegistersForReadInProject=0;resource=/Time Synchronization/Offset from Time Reference Valid;0;ReadMethodType=boolOffset from Time ReferenceNumberOfSyncRegistersForReadInProject=0;resource=/Time Synchronization/Offset from Time Reference;0;ReadMethodType=i32Register-B_2"DataType=1000800000000001003C005F03510020000000100001000100000010FFFFFFFFFFFFFFFF0000001F0000000F000000007FFFFFFF00000001FFFFFFF1000000000000000100010000000000000000000000000000;InitDataHash=C72400DF668E56426D1045502381E7C3;Name=Register-B_2;WriteArb=1"Reset RT Appresource=/Reset RT App;0;WriteMethodType=boolScan Clockresource=/Scan Clock;0;ReadMethodType=boolSleepresource=/Sleep;0;ReadMethodType=bool;WriteMethodType=boolSystem Resetresource=/System Reset;0;ReadMethodType=bool;WriteMethodType=boolSystem Watchdog ExpiredNumberOfSyncRegistersForReadInProject=Auto;resource=/System Watchdog Expired;0;ReadMethodType=boolTime SourceNumberOfSyncRegistersForReadInProject=0;resource=/Time Synchronization/Time Source;0;ReadMethodType=Targets\NI\FPGA\RIO\CompactRIO\Sync\SyncSource.ctlTime Synchronization FaultNumberOfSyncRegistersForReadInProject=0;resource=/Time Synchronization/Time Synchronization Fault;0;ReadMethodType=boolTimeNumberOfSyncRegistersForReadInProject=0;resource=/Time Synchronization/Time;0;ReadMethodType=u64USER FPGA LEDArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/USER FPGA LED;0;ReadMethodType=bool;WriteMethodType=bool</Property>
					<Property Name="NI.LV.FPGA.InterfaceBitfile" Type="Str">D:\github\PART-MLC\MLP\MLP-sizing\v1.2\FPGA Bitfiles\mlpsizing_FPGATarget_target_1fQeBnJI5tw.lvbitx</Property>
				</Item>
				<Item Name="FIFO-X" Type="FPGA FIFO">
					<Property Name="Actual Number of Elements" Type="UInt">4101</Property>
					<Property Name="Arbitration for Read" Type="UInt">1</Property>
					<Property Name="Arbitration for Write" Type="UInt">1</Property>
					<Property Name="Control Logic" Type="UInt">0</Property>
					<Property Name="Data Type" Type="UInt">9</Property>
					<Property Name="Disable on Overflow/Underflow" Type="Bool">false</Property>
					<Property Name="fifo.configuration" Type="Str">"ControlLogic=0;NumberOfElements=4101;Type=1;ReadArbs=Arbitrate if Multiple Requestors Only;ElementsPerRead=1;WriteArbs=Arbitrate if Multiple Requestors Only;ElementsPerWrite=1;Implementation=2;FIFO-X;DataType=1000800000000001003C005F03510020000000100001000100000010FFFFFFFFFFFFFFFF0000001F0000000F000000007FFFFFFF00000001FFFFFFF1000000000000000100010000000000000000000000000000;DisableOnOverflowUnderflow=FALSE"</Property>
					<Property Name="fifo.configured" Type="Bool">true</Property>
					<Property Name="fifo.projectItemValid" Type="Bool">true</Property>
					<Property Name="fifo.valid" Type="Bool">true</Property>
					<Property Name="fifo.version" Type="Int">13</Property>
					<Property Name="FPGA.PersistentID" Type="Str">{028C830A-A095-4346-A0A7-544E131465F0}</Property>
					<Property Name="Local" Type="Bool">false</Property>
					<Property Name="Memory Type" Type="UInt">2</Property>
					<Property Name="Number Of Elements Per Read" Type="UInt">1</Property>
					<Property Name="Number Of Elements Per Write" Type="UInt">1</Property>
					<Property Name="Requested Number of Elements" Type="UInt">4101</Property>
					<Property Name="Type" Type="UInt">1</Property>
					<Property Name="Type Descriptor" Type="Str">1000800000000001003C005F03510020000000100001000100000010FFFFFFFFFFFFFFFF0000001F0000000F000000007FFFFFFF00000001FFFFFFF1000000000000000100010000000000000000000000000000</Property>
				</Item>
				<Item Name="FIFO-y" Type="FPGA FIFO">
					<Property Name="Actual Number of Elements" Type="UInt">8191</Property>
					<Property Name="Arbitration for Read" Type="UInt">1</Property>
					<Property Name="Arbitration for Write" Type="UInt">1</Property>
					<Property Name="Control Logic" Type="UInt">0</Property>
					<Property Name="Data Type" Type="UInt">9</Property>
					<Property Name="Disable on Overflow/Underflow" Type="Bool">false</Property>
					<Property Name="fifo.configuration" Type="Str">"ControlLogic=0;NumberOfElements=8191;Type=2;ReadArbs=Arbitrate if Multiple Requestors Only;ElementsPerRead=1;WriteArbs=Arbitrate if Multiple Requestors Only;ElementsPerWrite=1;Implementation=2;FIFO-y;DataType=1000800000000001003C005F03510021000000110001002100000011FFFFFFFF00000000000100210000001100000000FFFFFFFF00000001FFFFFFF1000000000000000100010000000000000000000000000000;DisableOnOverflowUnderflow=FALSE"</Property>
					<Property Name="fifo.configured" Type="Bool">true</Property>
					<Property Name="fifo.projectItemValid" Type="Bool">true</Property>
					<Property Name="fifo.valid" Type="Bool">true</Property>
					<Property Name="fifo.version" Type="Int">13</Property>
					<Property Name="FPGA.PersistentID" Type="Str">{CDAE0FC8-C1CE-4640-B2F8-CEB2A616ED33}</Property>
					<Property Name="Local" Type="Bool">false</Property>
					<Property Name="Memory Type" Type="UInt">2</Property>
					<Property Name="Number Of Elements Per Read" Type="UInt">1</Property>
					<Property Name="Number Of Elements Per Write" Type="UInt">1</Property>
					<Property Name="Requested Number of Elements" Type="UInt">4101</Property>
					<Property Name="Type" Type="UInt">2</Property>
					<Property Name="Type Descriptor" Type="Str">1000800000000001003C005F03510021000000110001002100000011FFFFFFFF00000000000100210000001100000000FFFFFFFF00000001FFFFFFF1000000000000000100010000000000000000000000000000</Property>
				</Item>
				<Item Name="Memory-B_0" Type="FPGA Memory Block">
					<Property Name="FPGA.PersistentID" Type="Str">{01EDE284-F625-4DAF-A186-F48E839FF44E}</Property>
					<Property Name="fullEmulation" Type="Bool">false</Property>
					<Property Name="Memory Latency" Type="UInt">0</Property>
					<Property Name="Multiple Clock Domains" Type="Bool">false</Property>
					<Property Name="NI.LV.FPGA.CompileConfigString" Type="Str">Actual Number of Elements=40;ReadArbs=1;WriteArbs=1;Implementation=1;DataType=1000800000000001003C005F03510020000000100001000100000010FFFFFFFFFFFFFFFF0000001F0000000F000000007FFFFFFF00000001FFFFFFF1000000000000000100010000000000000000000000000000;InitDataHash=6230BE6EBA6DB1FA4A9CC9833459AFA7;DRAM Selection=;DRAM Max Outstanding Requests=64;DRAM Include Byte Enables=FALSE;DRAM Grant Time=50;Interface Configuration=Read A-Write B;Multiple Clock Domains=FALSE;Memory Latency=0</Property>
					<Property Name="NI.LV.FPGA.MEMORY.ActualNumberOfElements" Type="UInt">40</Property>
					<Property Name="NI.LV.FPGA.MEMORY.DataWidth" Type="UInt">9</Property>
					<Property Name="NI.LV.FPGA.MEMORY.DramIncludeByteEnables" Type="Bool">false</Property>
					<Property Name="NI.LV.FPGA.MEMORY.DramMaxOutstandingRequests" Type="Int">64</Property>
					<Property Name="NI.LV.FPGA.MEMORY.DramSelection" Type="Str"></Property>
					<Property Name="NI.LV.FPGA.MEMORY.Init" Type="Bool">true</Property>
					<Property Name="NI.LV.FPGA.MEMORY.InitData" Type="Str">00000000000000000000110101000100000000000000000000110011011010110000000000000000000100110000110011111111111111111110001101101001111111111111111111100100101100110000000000000000000000101101011100000000000000000100000101110011111111111111111111011100100011110000000000000000000001000000111011111111111111111110100101101011000000000000000000100011101010000000000000000000001000110110001011111111111111111111011000000111111111111111111111011001101011001111111111111111110010100010111111111111111111111101010111000011111111111111111111100001111010010000000000000000000010010100001000000000000000000010110001100110111111111111111111010010010001000000000000000000001101111111110000000000000000000001010101011100000000000000000000100010111101010000000000000000010000101101100011111111111111111111100100001001000000000000000000000011111000010000000000000000000110110111111011111111111111111111111011010000111111111111111111100001011110011111111111111111110001101101100011111111111111111110001110010010000000000000000000100101101101110000000000000000000011101000001100000000000000000010001111001001000000000000000000000010010000111111111111111111110111011110000100000000000000000001001111110110111111111111111111101011011101011111111111111111110000011100111000000000000000000011000010001000</Property>
					<Property Name="NI.LV.FPGA.MEMORY.InitVIPath" Type="Str">memory_load_B_0.vi</Property>
					<Property Name="NI.LV.FPGA.MEMORY.InterfaceAArbitration" Type="UInt">1</Property>
					<Property Name="NI.LV.FPGA.MEMORY.InterfaceBArbitration" Type="UInt">1</Property>
					<Property Name="NI.LV.FPGA.MEMORY.InterfaceConfig" Type="UInt">0</Property>
					<Property Name="NI.LV.FPGA.MEMORY.RequestedNumberOfElements" Type="UInt">40</Property>
					<Property Name="NI.LV.FPGA.MEMORY.Type" Type="UInt">1</Property>
					<Property Name="NI.LV.FPGA.ScriptConfigString" Type="Str">Actual Number of Elements=40;ReadArbs=1;WriteArbs=1;Implementation=1;DataType=1000800000000001003C005F03510020000000100001000100000010FFFFFFFFFFFFFFFF0000001F0000000F000000007FFFFFFF00000001FFFFFFF1000000000000000100010000000000000000000000000000;InitDataHash=6230BE6EBA6DB1FA4A9CC9833459AFA7;DRAM Selection=;DRAM Max Outstanding Requests=64;DRAM Include Byte Enables=FALSE;DRAM Grant Time=50;Interface Configuration=Read A-Write B;Multiple Clock Domains=FALSE;Memory Latency=0Persist Memory ValuesFALSE;</Property>
					<Property Name="NI.LV.FPGA.Valid" Type="Bool">true</Property>
					<Property Name="NI.LV.FPGA.Version" Type="Int">10</Property>
					<Property Name="Type Descriptor" Type="Str">1000800000000001003C005F03510020000000100001000100000010FFFFFFFFFFFFFFFF0000001F0000000F000000007FFFFFFF00000001FFFFFFF1000000000000000100010000000000000000000000000000</Property>
				</Item>
				<Item Name="Memory-B_1" Type="FPGA Memory Block">
					<Property Name="FPGA.PersistentID" Type="Str">{44D42E83-4F36-431E-81AA-998E426CB264}</Property>
					<Property Name="fullEmulation" Type="Bool">false</Property>
					<Property Name="Memory Latency" Type="UInt">0</Property>
					<Property Name="Multiple Clock Domains" Type="Bool">false</Property>
					<Property Name="NI.LV.FPGA.CompileConfigString" Type="Str">Actual Number of Elements=40;ReadArbs=1;WriteArbs=1;Implementation=1;DataType=1000800000000001003C005F03510020000000100001000100000010FFFFFFFFFFFFFFFF0000001F0000000F000000007FFFFFFF00000001FFFFFFF1000000000000000100010000000000000000000000000000;InitDataHash=56484364DDE23929BFC36AE6BF653A38;DRAM Selection=;DRAM Max Outstanding Requests=64;DRAM Include Byte Enables=FALSE;DRAM Grant Time=50;Interface Configuration=Read A-Write B;Multiple Clock Domains=FALSE;Memory Latency=0</Property>
					<Property Name="NI.LV.FPGA.MEMORY.ActualNumberOfElements" Type="UInt">40</Property>
					<Property Name="NI.LV.FPGA.MEMORY.DataWidth" Type="UInt">9</Property>
					<Property Name="NI.LV.FPGA.MEMORY.DramIncludeByteEnables" Type="Bool">false</Property>
					<Property Name="NI.LV.FPGA.MEMORY.DramMaxOutstandingRequests" Type="Int">64</Property>
					<Property Name="NI.LV.FPGA.MEMORY.DramSelection" Type="Str"></Property>
					<Property Name="NI.LV.FPGA.MEMORY.Init" Type="Bool">true</Property>
					<Property Name="NI.LV.FPGA.MEMORY.InitData" Type="Str">11111111111111111110111000111011000000000000000000100111100110100000000000000000000001110011001011111111111111111101100110010011000000000000000000100101001110001111111111111111110110101100111111111111111111111011100001101010111111111111111111011010010101000000000000000000010010011100001100000000000000000001010111100111111111111111111111010101011110010000000000000000000111101100000100000000000000000001110011000111000000000000000000010011101111001111111111111111101111011111101111111111111111111101101111110110000000000000000000110001010111100000000000000000000111000011100111111111111111111101000101101100111111111111111111100011110000100000000000000000000001011011000100000000000000000011001010001010111111111111111111011010010100011111111111111111110001010100100111111111111111111110110011100000000000000000000000110101001011101111111111111111111000111010010111111111111111111100110110100101111111111111111111000001101111001111111111111111111011110101110111111111111111111011101010111110000000000000000000001001111111000000000000000000000101001001001011111111111111111111101001110110000000000000000000100011110010101111111111111111111110110011001000000000000000000010000000111010111111111111111111011010001110110000000000000000001000101001110011111111111111111111111010011000</Property>
					<Property Name="NI.LV.FPGA.MEMORY.InitVIPath" Type="Str">memory_load_B_1.vi</Property>
					<Property Name="NI.LV.FPGA.MEMORY.InterfaceAArbitration" Type="UInt">1</Property>
					<Property Name="NI.LV.FPGA.MEMORY.InterfaceBArbitration" Type="UInt">1</Property>
					<Property Name="NI.LV.FPGA.MEMORY.InterfaceConfig" Type="UInt">0</Property>
					<Property Name="NI.LV.FPGA.MEMORY.RequestedNumberOfElements" Type="UInt">40</Property>
					<Property Name="NI.LV.FPGA.MEMORY.Type" Type="UInt">1</Property>
					<Property Name="NI.LV.FPGA.ScriptConfigString" Type="Str">Actual Number of Elements=40;ReadArbs=1;WriteArbs=1;Implementation=1;DataType=1000800000000001003C005F03510020000000100001000100000010FFFFFFFFFFFFFFFF0000001F0000000F000000007FFFFFFF00000001FFFFFFF1000000000000000100010000000000000000000000000000;InitDataHash=56484364DDE23929BFC36AE6BF653A38;DRAM Selection=;DRAM Max Outstanding Requests=64;DRAM Include Byte Enables=FALSE;DRAM Grant Time=50;Interface Configuration=Read A-Write B;Multiple Clock Domains=FALSE;Memory Latency=0Persist Memory ValuesFALSE;</Property>
					<Property Name="NI.LV.FPGA.Valid" Type="Bool">true</Property>
					<Property Name="NI.LV.FPGA.Version" Type="Int">10</Property>
					<Property Name="Type Descriptor" Type="Str">1000800000000001003C005F03510020000000100001000100000010FFFFFFFFFFFFFFFF0000001F0000000F000000007FFFFFFF00000001FFFFFFF1000000000000000100010000000000000000000000000000</Property>
				</Item>
				<Item Name="Register-B_2" Type="FPGA Register">
					<Property Name="Arbitration For Write" Type="UInt">1</Property>
					<Property Name="Compile Config String" Type="Str">"DataType=1000800000000001003C005F03510020000000100001000100000010FFFFFFFFFFFFFFFF0000001F0000000F000000007FFFFFFF00000001FFFFFFF1000000000000000100010000000000000000000000000000;InitDataHash=C72400DF668E56426D1045502381E7C3;Name=Register-B_2;WriteArb=1"</Property>
					<Property Name="Data Type" Type="UInt">9</Property>
					<Property Name="FPGA.PersistentID" Type="Str">{0108A0FA-EA5D-4E2E-9B5C-738854101D10}</Property>
					<Property Name="Initial Data" Type="Str">00000000000000000001110000001001</Property>
					<Property Name="Initialized" Type="Bool">true</Property>
					<Property Name="InitVIPath" Type="Str">memory_load_B_2.vi</Property>
					<Property Name="Type Descriptor" Type="Str">1000800000000001003C005F03510020000000100001000100000010FFFFFFFFFFFFFFFF0000001F0000000F000000007FFFFFFF00000001FFFFFFF1000000000000000100010000000000000000000000000000</Property>
					<Property Name="Valid" Type="Bool">true</Property>
					<Property Name="Version" Type="Int">1</Property>
				</Item>
				<Item Name="FIFO-out" Type="FPGA FIFO">
					<Property Name="Actual Number of Elements" Type="UInt">1023</Property>
					<Property Name="Arbitration for Read" Type="UInt">1</Property>
					<Property Name="Arbitration for Write" Type="UInt">1</Property>
					<Property Name="Control Logic" Type="UInt">0</Property>
					<Property Name="Data Type" Type="UInt">9</Property>
					<Property Name="Disable on Overflow/Underflow" Type="Bool">false</Property>
					<Property Name="fifo.configuration" Type="Str">"ControlLogic=0;NumberOfElements=1023;Type=2;ReadArbs=Arbitrate if Multiple Requestors Only;ElementsPerRead=1;WriteArbs=Arbitrate if Multiple Requestors Only;ElementsPerWrite=1;Implementation=2;FIFO-out;DataType=1000800000000001003C005F03510020000000100001002000000010FFFFFFFF800000000001002000000010000000007FFFFFFF00000001FFFFFFF1000000000000000100010000000000000000000000000000;DisableOnOverflowUnderflow=FALSE"</Property>
					<Property Name="fifo.configured" Type="Bool">true</Property>
					<Property Name="fifo.projectItemValid" Type="Bool">true</Property>
					<Property Name="fifo.valid" Type="Bool">true</Property>
					<Property Name="fifo.version" Type="Int">13</Property>
					<Property Name="FPGA.PersistentID" Type="Str">{9553AE3F-A318-4A94-872F-76C42FD27EA6}</Property>
					<Property Name="Local" Type="Bool">false</Property>
					<Property Name="Memory Type" Type="UInt">2</Property>
					<Property Name="Number Of Elements Per Read" Type="UInt">1</Property>
					<Property Name="Number Of Elements Per Write" Type="UInt">1</Property>
					<Property Name="Requested Number of Elements" Type="UInt">1023</Property>
					<Property Name="Type" Type="UInt">2</Property>
					<Property Name="Type Descriptor" Type="Str">1000800000000001003C005F03510020000000100001002000000010FFFFFFFF800000000001002000000010000000007FFFFFFF00000001FFFFFFF1000000000000000100010000000000000000000000000000</Property>
				</Item>
				<Item Name="FIFO address out" Type="FPGA FIFO">
					<Property Name="Actual Number of Elements" Type="UInt">1023</Property>
					<Property Name="Arbitration for Read" Type="UInt">1</Property>
					<Property Name="Arbitration for Write" Type="UInt">1</Property>
					<Property Name="Control Logic" Type="UInt">0</Property>
					<Property Name="Data Type" Type="UInt">3</Property>
					<Property Name="Disable on Overflow/Underflow" Type="Bool">false</Property>
					<Property Name="fifo.configuration" Type="Str">"ControlLogic=0;NumberOfElements=1023;Type=2;ReadArbs=Arbitrate if Multiple Requestors Only;ElementsPerRead=1;WriteArbs=Arbitrate if Multiple Requestors Only;ElementsPerWrite=1;Implementation=2;FIFO address out;DataType=1000800000000001000940030003493332000100000000000000000000;DisableOnOverflowUnderflow=FALSE"</Property>
					<Property Name="fifo.configured" Type="Bool">true</Property>
					<Property Name="fifo.projectItemValid" Type="Bool">true</Property>
					<Property Name="fifo.valid" Type="Bool">true</Property>
					<Property Name="fifo.version" Type="Int">13</Property>
					<Property Name="FPGA.PersistentID" Type="Str">{C6F6A12D-A136-4F8C-8F25-AC2358011335}</Property>
					<Property Name="Local" Type="Bool">false</Property>
					<Property Name="Memory Type" Type="UInt">2</Property>
					<Property Name="Number Of Elements Per Read" Type="UInt">1</Property>
					<Property Name="Number Of Elements Per Write" Type="UInt">1</Property>
					<Property Name="Requested Number of Elements" Type="UInt">1023</Property>
					<Property Name="Type" Type="UInt">2</Property>
					<Property Name="Type Descriptor" Type="Str">1000800000000001000940030003493332000100000000000000000000</Property>
				</Item>
				<Item Name="SubVI_bias.vi" Type="VI" URL="../../v1.1/SubVI_bias.vi">
					<Property Name="configString.guid" Type="Str">{0108A0FA-EA5D-4E2E-9B5C-738854101D10}"DataType=1000800000000001003C005F03510020000000100001000100000010FFFFFFFFFFFFFFFF0000001F0000000F000000007FFFFFFF00000001FFFFFFF1000000000000000100010000000000000000000000000000;InitDataHash=C72400DF668E56426D1045502381E7C3;Name=Register-B_2;WriteArb=1"{01EDE284-F625-4DAF-A186-F48E839FF44E}Actual Number of Elements=40;ReadArbs=1;WriteArbs=1;Implementation=1;DataType=1000800000000001003C005F03510020000000100001000100000010FFFFFFFFFFFFFFFF0000001F0000000F000000007FFFFFFF00000001FFFFFFF1000000000000000100010000000000000000000000000000;InitDataHash=6230BE6EBA6DB1FA4A9CC9833459AFA7;DRAM Selection=;DRAM Max Outstanding Requests=64;DRAM Include Byte Enables=FALSE;DRAM Grant Time=50;Interface Configuration=Read A-Write B;Multiple Clock Domains=FALSE;Memory Latency=0{028C830A-A095-4346-A0A7-544E131465F0}"ControlLogic=0;NumberOfElements=4101;Type=1;ReadArbs=Arbitrate if Multiple Requestors Only;ElementsPerRead=1;WriteArbs=Arbitrate if Multiple Requestors Only;ElementsPerWrite=1;Implementation=2;FIFO-X;DataType=1000800000000001003C005F03510020000000100001000100000010FFFFFFFFFFFFFFFF0000001F0000000F000000007FFFFFFF00000001FFFFFFF1000000000000000100010000000000000000000000000000;DisableOnOverflowUnderflow=FALSE"{0C5E1042-07A8-4DF0-AD6F-D3EDD24C1389}resource=/crio_Mod1/AI2;0;ReadMethodType=vi.lib\LabVIEW Targets\FPGA\cRIO\shared\nicrio_FXP_Controls\nicrio_FXP_S_24_7.ctl{12B5628B-C73D-4E38-946B-F07CC983EE81}NumberOfSyncRegistersForReadInProject=0;resource=/Time Synchronization/Time Synchronization Fault;0;ReadMethodType=bool{2E53299B-84AB-42B5-9E92-D537425137F4}NumberOfSyncRegistersForReadInProject=0;resource=/Time Synchronization/Time Source;0;ReadMethodType=Targets\NI\FPGA\RIO\CompactRIO\Sync\SyncSource.ctl{3798F8D1-9CEC-49B8-A086-0EF944D35F28}NumberOfSyncRegistersForReadInProject=Auto;resource=/cRIO_Trig/cRIO_Trig6;0;ReadMethodType=bool{3B253AA7-D9D6-45E4-B319-A7EF4E999194}NumberOfSyncRegistersForReadInProject=0;resource=/Time Synchronization/Offset from Time Reference;0;ReadMethodType=i32{44D42E83-4F36-431E-81AA-998E426CB264}Actual Number of Elements=40;ReadArbs=1;WriteArbs=1;Implementation=1;DataType=1000800000000001003C005F03510020000000100001000100000010FFFFFFFFFFFFFFFF0000001F0000000F000000007FFFFFFF00000001FFFFFFF1000000000000000100010000000000000000000000000000;InitDataHash=56484364DDE23929BFC36AE6BF653A38;DRAM Selection=;DRAM Max Outstanding Requests=64;DRAM Include Byte Enables=FALSE;DRAM Grant Time=50;Interface Configuration=Read A-Write B;Multiple Clock Domains=FALSE;Memory Latency=0{45B52DC7-1B6B-458C-BA29-D5BF8548F78B}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/USER FPGA LED;0;ReadMethodType=bool;WriteMethodType=bool{47D8E7DD-4CF9-432C-8810-ABB39830BC57}resource=/crio_Mod1/AI0;0;ReadMethodType=vi.lib\LabVIEW Targets\FPGA\cRIO\shared\nicrio_FXP_Controls\nicrio_FXP_S_24_7.ctl{606B95AA-A63E-4117-8D68-F7F4781E2EE4}NumberOfSyncRegistersForReadInProject=0;resource=/Time Synchronization/Offset from Time Reference Valid;0;ReadMethodType=bool{657EFC45-87EA-4AA9-B7F7-AA10B0EA39FF}NumberOfSyncRegistersForReadInProject=Auto;resource=/12.8 MHz Timebase;0;ReadMethodType=bool{6682DAFE-DE9E-4ED3-BCDB-D114C7198919}NumberOfSyncRegistersForReadInProject=Auto;resource=/10 MHz Timebase;0;ReadMethodType=bool{76FCE484-C688-4869-AB80-F6D1607E8C1F}[crioConfig.Begin]crio.Calibration=1,crio.Location=Slot 1,crio.Type=NI 9229,cRIOModule.ClockSource=1,cRIOModule.DataRate=1,cRIOModule.EnableDECoM=false,cRIOModule.EnableInputFifo=false,cRIOModule.EnableOutputFifo=false,cRIOModule.ExcitationVoltage=1,cRIOModule.ExternalClockSource=&lt;Carrier Clock 12.8 MHz&gt;,cRIOModule.ExtTimeBaseType=,cRIOModule.HalfBridgeEnable=0,cRIOModule.InputConfiguration=0,cRIOModule.RsiAttributes=,cRIOModule.SourceModule=false,cRIOModule.SubPanVisitedVersion=0,cRIOModule.TEDSSupport=false[crioConfig.End]{78EE6008-AB58-42A7-B18E-12F6A98CF225}NumberOfSyncRegistersForReadInProject=0;resource=/Time Synchronization/Time;0;ReadMethodType=u64{8FC19DEA-0D11-4599-9E28-DC65CA8AF064}resource=/System Reset;0;ReadMethodType=bool;WriteMethodType=bool{94F36DD7-2233-4FDE-8092-B11776C65DC4}resource=/Sleep;0;ReadMethodType=bool;WriteMethodType=bool{9553AE3F-A318-4A94-872F-76C42FD27EA6}"ControlLogic=0;NumberOfElements=1023;Type=2;ReadArbs=Arbitrate if Multiple Requestors Only;ElementsPerRead=1;WriteArbs=Arbitrate if Multiple Requestors Only;ElementsPerWrite=1;Implementation=2;FIFO-out;DataType=1000800000000001003C005F03510020000000100001002000000010FFFFFFFF800000000001002000000010000000007FFFFFFF00000001FFFFFFF1000000000000000100010000000000000000000000000000;DisableOnOverflowUnderflow=FALSE"{A903D566-0BD3-454C-95C0-B2EFE1604CC5}resource=/crio_Mod1/Start;0;WriteMethodType=bool{A91C6821-F05C-4670-89E6-A8E5CF47C64F}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/cRIO_Trig/cRIO_Trig3;0;ReadMethodType=bool;WriteMethodType=bool{B39D2A0E-92B9-4AFB-BF77-E37EC8B775D3}resource=/Chassis Temperature;0;ReadMethodType=i16{B657C088-1DAE-48D4-84CE-773611A2C706}resource=/crio_Mod1/AI1;0;ReadMethodType=vi.lib\LabVIEW Targets\FPGA\cRIO\shared\nicrio_FXP_Controls\nicrio_FXP_S_24_7.ctl{B751EED9-7B1B-4B94-B98E-55FE93761ABA}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/cRIO_Trig/cRIO_Trig0;0;ReadMethodType=bool;WriteMethodType=bool{B89C2D4F-C063-4E12-BE2F-32DF9101DF08}resource=/Reset RT App;0;WriteMethodType=bool{BC4AA9AF-18AD-4EFA-99C8-11FC64433606}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/cRIO_Trig/cRIO_Trig2;0;ReadMethodType=bool;WriteMethodType=bool{BDBC90D1-52EE-478C-BBB6-769CA0BA60E8}ResourceName=40 MHz Onboard Clock;TopSignalConnect=Clk40;ClockSignalName=Clk40;MinFreq=40000000.000000;MaxFreq=40000000.000000;VariableFreq=0;NomFreq=40000000.000000;PeakPeriodJitter=250.000000;MinDutyCycle=50.000000;MaxDutyCycle=50.000000;Accuracy=100.000000;RunTime=0;SpreadSpectrum=0;GenericDataHash=D41D8CD98F00B204E9800998ECF8427E;{C4081B4C-8388-44F9-A5D4-ECA53D59D49E}resource=/Scan Clock;0;ReadMethodType=bool{C6F6A12D-A136-4F8C-8F25-AC2358011335}"ControlLogic=0;NumberOfElements=1023;Type=2;ReadArbs=Arbitrate if Multiple Requestors Only;ElementsPerRead=1;WriteArbs=Arbitrate if Multiple Requestors Only;ElementsPerWrite=1;Implementation=2;FIFO address out;DataType=1000800000000001000940030003493332000100000000000000000000;DisableOnOverflowUnderflow=FALSE"{C7DA056B-75EE-429C-809B-E43689F3A198}ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/cRIO_Trig/cRIO_Trig1;0;ReadMethodType=bool;WriteMethodType=bool{C93351AA-B198-4983-96FC-1B9642BBF5D5}NumberOfSyncRegistersForReadInProject=Auto;resource=/13.1072 MHz Timebase;0;ReadMethodType=bool{CDAE0FC8-C1CE-4640-B2F8-CEB2A616ED33}"ControlLogic=0;NumberOfElements=8191;Type=2;ReadArbs=Arbitrate if Multiple Requestors Only;ElementsPerRead=1;WriteArbs=Arbitrate if Multiple Requestors Only;ElementsPerWrite=1;Implementation=2;FIFO-y;DataType=1000800000000001003C005F03510021000000110001002100000011FFFFFFFF00000000000100210000001100000000FFFFFFFF00000001FFFFFFF1000000000000000100010000000000000000000000000000;DisableOnOverflowUnderflow=FALSE"{DFB582B4-FA82-42D3-8101-319FEC5DE56C}resource=/crio_Mod1/AI3;0;ReadMethodType=vi.lib\LabVIEW Targets\FPGA\cRIO\shared\nicrio_FXP_Controls\nicrio_FXP_S_24_7.ctl{E20DC649-7E53-4246-9B2D-1FFB9788C051}NumberOfSyncRegistersForReadInProject=Auto;resource=/cRIO_Trig/cRIO_Trig5;0;ReadMethodType=bool{E7E11666-FA67-4229-B696-EA6753A11ADC}NumberOfSyncRegistersForReadInProject=Auto;resource=/cRIO_Trig/cRIO_Trig7;0;ReadMethodType=bool{E86FE605-9A95-4E86-8258-393844E416A7}resource=/crio_Mod1/Stop;0;WriteMethodType=bool{E9C65040-8873-4139-BCD5-30552906293D}NumberOfSyncRegistersForReadInProject=Auto;resource=/cRIO_Trig/cRIO_Trig4;0;ReadMethodType=bool{F71C576A-7CEE-4205-B0EB-CBC2F7B40E67}NumberOfSyncRegistersForReadInProject=Auto;resource=/System Watchdog Expired;0;ReadMethodType=boolcRIO-9054/Clk40/falsefalseFPGA_EXECUTION_MODEFPGA_TARGETFPGA_TARGET_CLASSCRIO_9054FPGA_TARGET_FAMILYARTIX7TARGET_TYPEFPGA</Property>
					<Property Name="configString.name" Type="Str">10 MHz TimebaseNumberOfSyncRegistersForReadInProject=Auto;resource=/10 MHz Timebase;0;ReadMethodType=bool12.8 MHz TimebaseNumberOfSyncRegistersForReadInProject=Auto;resource=/12.8 MHz Timebase;0;ReadMethodType=bool13.1072 MHz TimebaseNumberOfSyncRegistersForReadInProject=Auto;resource=/13.1072 MHz Timebase;0;ReadMethodType=bool40 MHz Onboard ClockResourceName=40 MHz Onboard Clock;TopSignalConnect=Clk40;ClockSignalName=Clk40;MinFreq=40000000.000000;MaxFreq=40000000.000000;VariableFreq=0;NomFreq=40000000.000000;PeakPeriodJitter=250.000000;MinDutyCycle=50.000000;MaxDutyCycle=50.000000;Accuracy=100.000000;RunTime=0;SpreadSpectrum=0;GenericDataHash=D41D8CD98F00B204E9800998ECF8427E;Chassis Temperatureresource=/Chassis Temperature;0;ReadMethodType=i16cRIO_Trig0ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/cRIO_Trig/cRIO_Trig0;0;ReadMethodType=bool;WriteMethodType=boolcRIO_Trig1ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/cRIO_Trig/cRIO_Trig1;0;ReadMethodType=bool;WriteMethodType=boolcRIO_Trig2ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/cRIO_Trig/cRIO_Trig2;0;ReadMethodType=bool;WriteMethodType=boolcRIO_Trig3ArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/cRIO_Trig/cRIO_Trig3;0;ReadMethodType=bool;WriteMethodType=boolcRIO_Trig4NumberOfSyncRegistersForReadInProject=Auto;resource=/cRIO_Trig/cRIO_Trig4;0;ReadMethodType=boolcRIO_Trig5NumberOfSyncRegistersForReadInProject=Auto;resource=/cRIO_Trig/cRIO_Trig5;0;ReadMethodType=boolcRIO_Trig6NumberOfSyncRegistersForReadInProject=Auto;resource=/cRIO_Trig/cRIO_Trig6;0;ReadMethodType=boolcRIO_Trig7NumberOfSyncRegistersForReadInProject=Auto;resource=/cRIO_Trig/cRIO_Trig7;0;ReadMethodType=boolcRIO-9054/Clk40/falsefalseFPGA_EXECUTION_MODEFPGA_TARGETFPGA_TARGET_CLASSCRIO_9054FPGA_TARGET_FAMILYARTIX7TARGET_TYPEFPGAFIFO address out"ControlLogic=0;NumberOfElements=1023;Type=2;ReadArbs=Arbitrate if Multiple Requestors Only;ElementsPerRead=1;WriteArbs=Arbitrate if Multiple Requestors Only;ElementsPerWrite=1;Implementation=2;FIFO address out;DataType=1000800000000001000940030003493332000100000000000000000000;DisableOnOverflowUnderflow=FALSE"FIFO-out"ControlLogic=0;NumberOfElements=1023;Type=2;ReadArbs=Arbitrate if Multiple Requestors Only;ElementsPerRead=1;WriteArbs=Arbitrate if Multiple Requestors Only;ElementsPerWrite=1;Implementation=2;FIFO-out;DataType=1000800000000001003C005F03510020000000100001002000000010FFFFFFFF800000000001002000000010000000007FFFFFFF00000001FFFFFFF1000000000000000100010000000000000000000000000000;DisableOnOverflowUnderflow=FALSE"FIFO-X"ControlLogic=0;NumberOfElements=4101;Type=1;ReadArbs=Arbitrate if Multiple Requestors Only;ElementsPerRead=1;WriteArbs=Arbitrate if Multiple Requestors Only;ElementsPerWrite=1;Implementation=2;FIFO-X;DataType=1000800000000001003C005F03510020000000100001000100000010FFFFFFFFFFFFFFFF0000001F0000000F000000007FFFFFFF00000001FFFFFFF1000000000000000100010000000000000000000000000000;DisableOnOverflowUnderflow=FALSE"FIFO-y"ControlLogic=0;NumberOfElements=8191;Type=2;ReadArbs=Arbitrate if Multiple Requestors Only;ElementsPerRead=1;WriteArbs=Arbitrate if Multiple Requestors Only;ElementsPerWrite=1;Implementation=2;FIFO-y;DataType=1000800000000001003C005F03510021000000110001002100000011FFFFFFFF00000000000100210000001100000000FFFFFFFF00000001FFFFFFF1000000000000000100010000000000000000000000000000;DisableOnOverflowUnderflow=FALSE"Memory-B_0Actual Number of Elements=40;ReadArbs=1;WriteArbs=1;Implementation=1;DataType=1000800000000001003C005F03510020000000100001000100000010FFFFFFFFFFFFFFFF0000001F0000000F000000007FFFFFFF00000001FFFFFFF1000000000000000100010000000000000000000000000000;InitDataHash=6230BE6EBA6DB1FA4A9CC9833459AFA7;DRAM Selection=;DRAM Max Outstanding Requests=64;DRAM Include Byte Enables=FALSE;DRAM Grant Time=50;Interface Configuration=Read A-Write B;Multiple Clock Domains=FALSE;Memory Latency=0Memory-B_1Actual Number of Elements=40;ReadArbs=1;WriteArbs=1;Implementation=1;DataType=1000800000000001003C005F03510020000000100001000100000010FFFFFFFFFFFFFFFF0000001F0000000F000000007FFFFFFF00000001FFFFFFF1000000000000000100010000000000000000000000000000;InitDataHash=56484364DDE23929BFC36AE6BF653A38;DRAM Selection=;DRAM Max Outstanding Requests=64;DRAM Include Byte Enables=FALSE;DRAM Grant Time=50;Interface Configuration=Read A-Write B;Multiple Clock Domains=FALSE;Memory Latency=0Mod1/AI0resource=/crio_Mod1/AI0;0;ReadMethodType=vi.lib\LabVIEW Targets\FPGA\cRIO\shared\nicrio_FXP_Controls\nicrio_FXP_S_24_7.ctlMod1/AI1resource=/crio_Mod1/AI1;0;ReadMethodType=vi.lib\LabVIEW Targets\FPGA\cRIO\shared\nicrio_FXP_Controls\nicrio_FXP_S_24_7.ctlMod1/AI2resource=/crio_Mod1/AI2;0;ReadMethodType=vi.lib\LabVIEW Targets\FPGA\cRIO\shared\nicrio_FXP_Controls\nicrio_FXP_S_24_7.ctlMod1/AI3resource=/crio_Mod1/AI3;0;ReadMethodType=vi.lib\LabVIEW Targets\FPGA\cRIO\shared\nicrio_FXP_Controls\nicrio_FXP_S_24_7.ctlMod1/Startresource=/crio_Mod1/Start;0;WriteMethodType=boolMod1/Stopresource=/crio_Mod1/Stop;0;WriteMethodType=boolMod1[crioConfig.Begin]crio.Calibration=1,crio.Location=Slot 1,crio.Type=NI 9229,cRIOModule.ClockSource=1,cRIOModule.DataRate=1,cRIOModule.EnableDECoM=false,cRIOModule.EnableInputFifo=false,cRIOModule.EnableOutputFifo=false,cRIOModule.ExcitationVoltage=1,cRIOModule.ExternalClockSource=&lt;Carrier Clock 12.8 MHz&gt;,cRIOModule.ExtTimeBaseType=,cRIOModule.HalfBridgeEnable=0,cRIOModule.InputConfiguration=0,cRIOModule.RsiAttributes=,cRIOModule.SourceModule=false,cRIOModule.SubPanVisitedVersion=0,cRIOModule.TEDSSupport=false[crioConfig.End]Offset from Time Reference ValidNumberOfSyncRegistersForReadInProject=0;resource=/Time Synchronization/Offset from Time Reference Valid;0;ReadMethodType=boolOffset from Time ReferenceNumberOfSyncRegistersForReadInProject=0;resource=/Time Synchronization/Offset from Time Reference;0;ReadMethodType=i32Register-B_2"DataType=1000800000000001003C005F03510020000000100001000100000010FFFFFFFFFFFFFFFF0000001F0000000F000000007FFFFFFF00000001FFFFFFF1000000000000000100010000000000000000000000000000;InitDataHash=C72400DF668E56426D1045502381E7C3;Name=Register-B_2;WriteArb=1"Reset RT Appresource=/Reset RT App;0;WriteMethodType=boolScan Clockresource=/Scan Clock;0;ReadMethodType=boolSleepresource=/Sleep;0;ReadMethodType=bool;WriteMethodType=boolSystem Resetresource=/System Reset;0;ReadMethodType=bool;WriteMethodType=boolSystem Watchdog ExpiredNumberOfSyncRegistersForReadInProject=Auto;resource=/System Watchdog Expired;0;ReadMethodType=boolTime SourceNumberOfSyncRegistersForReadInProject=0;resource=/Time Synchronization/Time Source;0;ReadMethodType=Targets\NI\FPGA\RIO\CompactRIO\Sync\SyncSource.ctlTime Synchronization FaultNumberOfSyncRegistersForReadInProject=0;resource=/Time Synchronization/Time Synchronization Fault;0;ReadMethodType=boolTimeNumberOfSyncRegistersForReadInProject=0;resource=/Time Synchronization/Time;0;ReadMethodType=u64USER FPGA LEDArbitrationForOutputData=NeverArbitrate;NumberOfSyncRegistersForOutputData=1;NumberOfSyncRegistersForReadInProject=Auto;resource=/USER FPGA LED;0;ReadMethodType=bool;WriteMethodType=bool</Property>
				</Item>
				<Item Name="Dependencies" Type="Dependencies">
					<Item Name="vi.lib" Type="Folder">
						<Item Name="Clear Errors.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/Clear Errors.vi"/>
						<Item Name="LVFixedPointQuantizationPolicyTypeDef.ctl" Type="VI" URL="/&lt;vilib&gt;/fxp/LVFixedPointQuantizationPolicyTypeDef.ctl"/>
						<Item Name="LVFixedPointOverflowPolicyTypeDef.ctl" Type="VI" URL="/&lt;vilib&gt;/fxp/LVFixedPointOverflowPolicyTypeDef.ctl"/>
						<Item Name="FxpSim.dll" Type="Document" URL="/&lt;vilib&gt;/rvi/FXPMathLib/sim/FxpSim.dll"/>
						<Item Name="lvSimController.dll" Type="Document" URL="/&lt;vilib&gt;/rvi/Simulation/lvSimController.dll"/>
					</Item>
				</Item>
				<Item Name="Build Specifications" Type="Build">
					<Item Name="target" Type="{F4C5E96F-7410-48A5-BB87-3559BC9B167F}">
						<Property Name="AllowEnableRemoval" Type="Bool">false</Property>
						<Property Name="BuildSpecDecription" Type="Str"></Property>
						<Property Name="BuildSpecName" Type="Str">target</Property>
						<Property Name="Comp.BitfileName" Type="Str">matrixmultiply_FPGATarget_target_oi-7vW+psaw.lvbitx</Property>
						<Property Name="Comp.CustomXilinxParameters" Type="Str"></Property>
						<Property Name="Comp.MaxFanout" Type="Int">-1</Property>
						<Property Name="Comp.RandomSeed" Type="Bool">false</Property>
						<Property Name="Comp.Version.Build" Type="Int">0</Property>
						<Property Name="Comp.Version.Fix" Type="Int">0</Property>
						<Property Name="Comp.Version.Major" Type="Int">1</Property>
						<Property Name="Comp.Version.Minor" Type="Int">0</Property>
						<Property Name="Comp.VersionAutoIncrement" Type="Bool">false</Property>
						<Property Name="Comp.Vivado.EnableMultiThreading" Type="Bool">true</Property>
						<Property Name="Comp.Vivado.OptDirective" Type="Str"></Property>
						<Property Name="Comp.Vivado.PhysOptDirective" Type="Str"></Property>
						<Property Name="Comp.Vivado.PlaceDirective" Type="Str"></Property>
						<Property Name="Comp.Vivado.RouteDirective" Type="Str"></Property>
						<Property Name="Comp.Vivado.RunPowerOpt" Type="Bool">false</Property>
						<Property Name="Comp.Vivado.Strategy" Type="Str">Default</Property>
						<Property Name="Comp.Xilinx.DesignStrategy" Type="Str">balanced</Property>
						<Property Name="Comp.Xilinx.MapEffort" Type="Str">default(noTiming)</Property>
						<Property Name="Comp.Xilinx.ParEffort" Type="Str">standard</Property>
						<Property Name="Comp.Xilinx.SynthEffort" Type="Str">normal</Property>
						<Property Name="Comp.Xilinx.SynthGoal" Type="Str">speed</Property>
						<Property Name="Comp.Xilinx.UseRecommended" Type="Bool">true</Property>
						<Property Name="DefaultBuildSpec" Type="Bool">true</Property>
						<Property Name="DestinationDirectory" Type="Path">FPGA Bitfiles</Property>
						<Property Name="NI.LV.FPGA.LastCompiledBitfilePath" Type="Path">/D/github/PART-MLC/MLP/MLP-sizing/v1.2/FPGA Bitfiles/mlpsizing_FPGATarget_target_1fQeBnJI5tw.lvbitx</Property>
						<Property Name="NI.LV.FPGA.LastCompiledBitfilePathRelativeToProject" Type="Path">FPGA Bitfiles/mlpsizing_FPGATarget_target_1fQeBnJI5tw.lvbitx</Property>
						<Property Name="ProjectPath" Type="Path">/C/Users/adowney2/Documents/attention_layer_algorithm/software/MLP/matrix_multiply/matrix_multiply.lvproj</Property>
						<Property Name="RelativePath" Type="Bool">true</Property>
						<Property Name="RunWhenLoaded" Type="Bool">false</Property>
						<Property Name="SupportDownload" Type="Bool">true</Property>
						<Property Name="SupportResourceEstimation" Type="Bool">false</Property>
						<Property Name="TargetName" Type="Str">FPGA Target</Property>
						<Property Name="TopLevelVI" Type="Ref">/NI-cRIO-9054-01F17EC3/Chassis/FPGA Target/target.vi</Property>
					</Item>
				</Item>
			</Item>
		</Item>
		<Item Name="Dependencies" Type="Dependencies"/>
		<Item Name="Build Specifications" Type="Build"/>
	</Item>
</Project>
