$condaBat = 'C:\Users\admin\miniconda3\condabin\conda.bat'
$envYml = 'C:\Users\admin\Documents\GitHub\proyecto-machinelearning-telco\environment.yml'

Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned -Force
& $condaBat init powershell

$envList = (& $condaBat env list 2>&1 | Out-String)
Write-Output $envList

if ($envList -notmatch 'telco-ml') {
    Write-Output 'Creando entorno telco-ml...'
    & $condaBat env create -f $envYml
} else {
    Write-Output 'Actualizando entorno telco-ml...'
    & $condaBat env update -f $envYml --prune
}
