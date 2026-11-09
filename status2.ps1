$pw = ConvertTo-SecureString "kczx05" -AsPlainText -Force
$cred = New-Object System.Management.Automation.PSCredential("Administrator", $pw)
$so = New-PSSessionOption -OperationTimeout 120000
$sb = {
  Get-CimInstance Win32_Process | Where-Object { $_.Name -match "clang|icpx|cl|link|cmd|bash|powershell" } |
    Sort-Object -Property WorkingSetSize -Descending | Select-Object -First 6 Name,ProcessId,
    @{n="cmdline";e={$_.CommandLine.Substring(0,[Math]::Min(110,$_.CommandLine.Length))}} | Format-List | Out-String
  "--- C:\nova ---"
  Get-ChildItem C:\nova | Select-Object -ExpandProperty Name
  "--- build ---"
  if (Test-Path C:\nova\build) { Get-ChildItem C:\nova\build -Recurse -Depth 1 | Select-Object FullName,Length | Format-Table -AutoSize | Out-String } else { "no build dir" }
}
Invoke-Command -ComputerName 192.168.137.8 -Credential $cred -ScriptBlock $sb -SessionOption $so | Out-String -Width 200