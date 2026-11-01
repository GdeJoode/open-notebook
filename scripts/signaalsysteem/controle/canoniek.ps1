# Canoniek pad op de host voor de startcontrole (F2-10, padlijsten.py).
#
# Leest paden van stdin (een per regel, UTF-8) en schrijft JSON
# {"pad": "canoniek pad" | null}. Lost op: de vormen \\?\D:\, /run/desktop/mnt/host/d/
# en /mnt/d/, subst-stations, junctions en symbolische koppelingen (ook halverwege
# het pad) en 8.3-namen. Een pad dat niet bestaat, of een segment "..", ".", of met
# een jokerteken, geeft null: de startcontrole weigert dan.
$ErrorActionPreference = 'Stop'
[Console]::InputEncoding = [Text.Encoding]::UTF8
[Console]::OutputEncoding = [Text.Encoding]::UTF8

$subst = @{}
foreach ($regel in @(subst.exe)) {
    if ($regel -match '^([A-Za-z]):\\: => (.+)$') { $subst[$Matches[1].ToUpper()] = $Matches[2] }
}

function Canoniek([string]$pad, [int]$diepte) {
    if ($diepte -gt 16) { return $null }
    $p = $pad -replace '/', '\'
    $p = $p -replace '^\\\\\?\\', '' -replace '^\\\\\.\\', '' -replace '^\\\?\?\\', ''
    if ($p -match '^\\run\\desktop\\mnt\\host\\([A-Za-z])(\\.*)?$') { $p = $Matches[1] + ':' + $Matches[2] }
    elseif ($p -match '^\\mnt\\([A-Za-z])(\\.*)?$') { $p = $Matches[1] + ':' + $Matches[2] }
    if ($p -notmatch '^[A-Za-z]:(\\|$)') { return $null }
    $letter = $p.Substring(0, 1).ToUpper()
    $rest = if ($p.Length -gt 3) { $p.Substring(3) } else { '' }
    if ($subst.ContainsKey($letter)) {
        return Canoniek ($subst[$letter].TrimEnd('\') + '\' + $rest) ($diepte + 1)
    }
    $delen = @($rest -split '\\' | Where-Object { $_ -ne '' })
    $huidig = $letter + ':\'
    for ($i = 0; $i -lt $delen.Count; $i++) {
        $deel = $delen[$i]
        if ($deel -eq '..' -or $deel -eq '.' -or $deel -match '[*?]') { return $null }
        # -Filter gebruikt de Windows-zoekfunctie: vindt ook een 8.3-naam en geeft de lange naam.
        $item = Get-ChildItem -LiteralPath $huidig -Force -Filter $deel -ErrorAction SilentlyContinue |
            Select-Object -First 1
        if (-not $item) { return $null }
        $volgende = Join-Path $huidig $item.Name
        if ($item.LinkType) {
            $doel = (@($item.Target)[0]) -replace '^\\\?\?\\', ''
            if (-not [IO.Path]::IsPathRooted($doel)) { $doel = Join-Path $huidig $doel }
            $over = if ($i + 1 -lt $delen.Count) { '\' + ($delen[($i + 1)..($delen.Count - 1)] -join '\') } else { '' }
            return Canoniek ($doel + $over) ($diepte + 1)
        }
        $huidig = $volgende
    }
    return ($huidig -replace '\\', '/')
}

$uit = [ordered]@{}
foreach ($regel in [Console]::In.ReadToEnd() -split "`r?`n") {
    if ($regel -ne '') { $uit[$regel] = Canoniek $regel 0 }
}
$uit | ConvertTo-Json -Compress
