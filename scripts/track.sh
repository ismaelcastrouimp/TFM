#!/bin/bash
# track.sh: variante de TFM de ~/Documents/tools/jobmon/track que registra en TFM/.jobs (la sesión de nono no puede
# escribir en ~/.local/state/jobmon).  Lanza un comando registrándolo para jobmon-gui; al terminar pasa al historial.
# El comando se ejecuta desde la raíz de TFM (las rutas relativas son relativas a TFM).
# Uso:  scripts/track.sh [-l <log>] <etiqueta> <comando...>
#   sin -l: log TEMPORAL en .jobs/logs/<etiqueta>-<fecha>.log ("Borrar historial" en jobmon-gui lo borra).
#           Para pruebas, depuración, mediciones de velocidad.
#   con -l: log PERMANENTE en <log> (p. ej. data/<experimento>/<etiqueta>.log; se sobrescribe si existe).
#           Para todo resultado que se cite en progreso.md, se compare o dibuje, o sea caro de repetir.
#   Forma antigua (sigue valiendo): scripts/track.sh <etiqueta> <log.log> <comando...>  == -l <log.log>
# NO reescribir este fichero en sitio con trabajos en marcha (escribir aparte y mv).
usage() { echo "uso: scripts/track.sh [-l <log>] <etiqueta> <comando...>"; exit 1; }
cd "$(dirname "$0")/.."; D=${JOBMON_DIR:-$PWD/.jobs}
log=
if [ "$1" = "-l" ]; then [ $# -ge 2 ] || usage; log=$(realpath -m "$2"); shift 2
elif [ $# -ge 3 ] && [[ "$2" == *.log ]]; then log=$(realpath -m "$2"); set -- "$1" "${@:3}"; fi
[ $# -lt 2 ] && usage
label=$1; shift
[ -n "$log" ] || log="$D/logs/$label-$(date +%Y%m%d-%H%M%S).log"
mkdir -p "$D" "$(dirname "$log")"
f="$D/$label.job"; start=$(date +%s)
argv=$(printf '%q ' "$@")   # el comando con sus comillas, para poder relanzarlo tal cual
printf 'pid=%s\nstart=%s\nlog=%s\ncwd=%s\ncmd=%s\nargv=%s\n' "$$" "$start" "$log" "$PWD" "$*" "$argv" > "$f"
js() { local s=${1//\\/\\\\}; s=${s//\"/\\\"}; printf '"%s"' "${s//$'\n'/ }"; }
finish() {
  st=${st:-130}
  printf '{"label":%s,"start":%s,"end":%s,"exit":%s,"log":%s,"cwd":%s,"cmd":%s,"argv":%s}\n' "$(js "$label")" \
    "$start" "$(date +%s)" "$st" "$(js "$log")" "$(js "$PWD")" "$(js "$cmd")" "$(js "$argv")" >> "$D/history.jsonl"
  rm -f "$f"
}
cmd="$*"
trap finish EXIT
trap 'st=143; exit 143' TERM INT
"$@" > "$log" 2>&1
st=$?
exit $st
