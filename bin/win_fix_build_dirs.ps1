Get-Location
$module_files = Get-ChildItem -Path . -Filter lightning_qubit_ops.* -Recurse
echo $module_files
ForEach-Object{$module_files} | Move-Item -Destination { $_.Directory.Parent.FullName }
dir