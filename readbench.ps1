$pw = ConvertTo-SecureString "kczx05" -AsPlainText -Force
$cred = New-Object System.Management.Automation.PSCredential("Administrator", $pw)
$so = New-PSSessionOption -OperationTimeout 120000
Invoke-Command -ComputerName 192.168.137.8 -Credential $cred -ScriptBlock {
  if (Test-Path C:\nova\bench_out.txt) { (Get-Content C:\nova\bench_out.txt -Tail 30 | Out-String) } else { "no log" }
} -SessionOption $so | Out-String -Width 200