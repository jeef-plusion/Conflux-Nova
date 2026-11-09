$pw = ConvertTo-SecureString "kczx05" -AsPlainText -Force
$cred = New-Object System.Management.Automation.PSCredential("Administrator", $pw)
$so = New-PSSessionOption -OperationTimeout 90000
Invoke-Command -ComputerName 192.168.137.8 -Credential $cred -ScriptBlock {
  if (Test-Path C:\nova\bench_out.txt) { "size=" + (Get-Item C:\nova\bench_out.txt).Length; (Get-Content C:\nova\bench_out.txt -Tail 35 | Out-String) } else { "no log yet" }
  "bench proc: " + ((Get-Process sycl_bench -ErrorAction SilentlyContinue | Measure-Object).Count)
} -SessionOption $so | Out-String -Width 200