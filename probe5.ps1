$pw = ConvertTo-SecureString "kczx05" -AsPlainText -Force
$cred = New-Object System.Management.Automation.PSCredential("Administrator", $pw)
$sb = {
  $ProgressPreference = "SilentlyContinue"
  try { $r = Invoke-WebRequest "https://www.intel.com" -UseBasicParsing -TimeoutSec 12; "intel http=" + $r.StatusCode } catch { "intel http=FAIL " + $_.Exception.Message }
  try { $r2 = Invoke-WebRequest "https://github.com" -UseBasicParsing -TimeoutSec 12; "github http=" + $r2.StatusCode } catch { "github http=FAIL" }
  (winget source list 2>&1 | Out-String)
}
Invoke-Command -ComputerName 192.168.137.8 -Credential $cred -ScriptBlock $sb -ErrorAction Stop