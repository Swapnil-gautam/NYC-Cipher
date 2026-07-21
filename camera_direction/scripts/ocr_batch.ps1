Add-Type -AssemblyName System.Drawing
Add-Type -AssemblyName System.Runtime.WindowsRuntime
$null = [Windows.Media.Ocr.OcrEngine, Windows.Foundation, ContentType=WindowsRuntime]
$null = [Windows.Storage.StorageFile, Windows.Storage, ContentType=WindowsRuntime]
$null = [Windows.Graphics.Imaging.BitmapDecoder, Windows.Graphics.Imaging, ContentType=WindowsRuntime]

Function Await($WinRtTask, $ResultType) {
    $asTask = ([System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object { $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1' })[0]
    $asTaskGeneric = $asTask.MakeGenericMethod($ResultType)
    $netTask = $asTaskGeneric.Invoke($null, @($WinRtTask))
    $netTask.Wait(-1) | Out-Null
    $netTask.Result
}
function OcrFile([string]$path) {
  $file = Await ([Windows.Storage.StorageFile]::GetFileFromPathAsync($path)) ([Windows.Storage.StorageFile])
  $stream = Await ($file.OpenAsync([Windows.Storage.FileAccessMode]::Read)) ([Windows.Storage.Streams.IRandomAccessStream])
  $decoder = Await ([Windows.Graphics.Imaging.BitmapDecoder]::CreateAsync($stream)) ([Windows.Graphics.Imaging.BitmapDecoder])
  $bitmap = Await ($decoder.GetSoftwareBitmapAsync()) ([Windows.Graphics.Imaging.SoftwareBitmap])
  $ocrEngine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromUserProfileLanguages()
  $result = Await ($ocrEngine.RecognizeAsync($bitmap)) ([Windows.Media.Ocr.OcrResult])
  $stream.Dispose()
  return $result.Text
}
function ParseDirection([string]$text) {
  $u = $text.ToUpper()
  if ($u -match 'NORTH') { return 'N' }
  if ($u -match 'SOUTH') { return 'S' }
  if ($u -match 'EAST')  { return 'E' }
  if ($u -match 'WEST')  { return 'W' }
  if ($u -match 'ORTH')  { return 'N' }
  if ($u -match 'OUTH')  { return 'S' }
  if ($u -match 'AST')   { return 'E' }
  if ($u -match 'EST')   { return 'W' }
  return $null
}

$scratch = "C:\Users\Acer\AppData\Local\Temp\claude\e--Projects-NYC-Data\699f97f2-e23f-41fd-83c2-674c88a90019\scratchpad"
$cams = Get-Content "$scratch\manhattan_cams.json" -Raw | ConvertFrom-Json

Add-Type -AssemblyName System.Drawing
[int]$cropX=0; [int]$cropY=0; [int]$cropW=130; [int]$cropH=20; [int]$scale=6
[int]$upW = $cropW * $scale
[int]$upH = $cropH * $scale
$tmpJpg = "$scratch\_ocr_tmp.jpg"
$tmpPng = "$scratch\_ocr_tmp.png"

$results = New-Object System.Collections.Generic.List[object]
$i = 0
foreach ($cam in $cams) {
  $i++
  try {
    Invoke-WebRequest -Uri $cam.url -OutFile $tmpJpg -TimeoutSec 8
    $img = [System.Drawing.Bitmap]::FromFile($tmpJpg)
    $srcRect = New-Object System.Drawing.Rectangle -ArgumentList $cropX, $cropY, $cropW, $cropH
    $bmp = New-Object System.Drawing.Bitmap -ArgumentList $upW, $upH
    $g = [System.Drawing.Graphics]::FromImage($bmp)
    $g.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
    $destRect = New-Object System.Drawing.Rectangle -ArgumentList 0, 0, $upW, $upH
    $g.DrawImage($img, $destRect, $srcRect, [System.Drawing.GraphicsUnit]::Pixel)
    $bmp.Save($tmpPng, [System.Drawing.Imaging.ImageFormat]::Png)
    $g.Dispose(); $bmp.Dispose(); $img.Dispose()

    $text = OcrFile $tmpPng
    $dir = ParseDirection $text
    $results.Add([PSCustomObject]@{ id = $cam.id; name = $cam.name; direction = $dir; ocrText = $text })
  } catch {
    $results.Add([PSCustomObject]@{ id = $cam.id; name = $cam.name; direction = $null; ocrText = "ERROR: $_" })
  }
  if ($i % 25 -eq 0) { Write-Host "Processed $i / $($cams.Count)" }
}

$results | ConvertTo-Json -Depth 3 | Out-File -Encoding utf8 "$scratch\manhattan_directions.json"
$withDir = ($results | Where-Object { $_.direction }).Count
Write-Host "DONE. Total: $($results.Count), with direction: $withDir"
