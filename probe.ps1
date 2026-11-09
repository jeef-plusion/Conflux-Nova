$ErrorActionPreference = "Stop"
Start-Service WinRM -ErrorAction SilentlyContinue
Set-Item WSMan:\localhost\Client\TrustedHosts -Value "192.168.137.8" -Force
$pw = ConvertTo-SecureString "kczx05" -AsPlainText -Force
$cred = New-Object System.Management.Automation.PSCredential("Administrator", $pw)
$sb = {
  "=== OS ==="
  (Get-CimInstance Win32_OperatingSystem).Caption + " build " + (Get-CimInstance Win32_OperatingSystem).BuildNumber
  "=== CPU ==="
  (Get-CimInstance Win32_Processor).Name
  "=== GPU ==="
  Get-CimInstance Win32_VideoController | ForEach-Object { $_.Name + " | driver " + $_.DriverVersion }
  "=== RAM(GB) ==="
  [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory/1GB,1)
  "=== DISK ==="
  Get-PSDrive -PSProvider FileSystem | Where-Object {$_.Used -ne $null} | ForEach-Object { $_.Name + ": free " + [math]::Round($_.Free/1GB,1) + " GB" }
  "=== TOOLS ==="
  foreach ($t in "nvcc","icpx","cl","cmake","git") { $c = Get-Command $t -ErrorAction SilentlyContinue; if ($c) { "$t -> " + $c.Source } else { "$t -> none" } }
  "=== HOSTNAME ==="
  $env:COMPUTERNAME
}
Invoke-Command -ComputerName 192.168.137.8 -Credential $cred -ScriptBlock $sb -ErrorAction Stop | Out-String