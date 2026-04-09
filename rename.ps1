Get-ChildItem -Directory | Where-Object { $_.Name -match '^course' } | ForEach-Object {
    if ($_.Name -match '^course-(\d)( |$)') {
        $num = $matches[1]
        $newName = $_.Name -replace "^course-(\d)", "course-0$num"
        Rename-Item -Path $_.FullName -NewName $newName
        Write-Host "Renamed: $($_.Name) -> $newName"
    }
}