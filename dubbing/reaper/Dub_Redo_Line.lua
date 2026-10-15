-- @description Dub: Redo this line (open the Redo-line popup)
-- @about
--   v0.29.14. Give it a keyboard shortcut with the panel's "Set a shortcut…"
--   button (Tools -> Redo one line, or the popup itself). Asks the running
--   Dub Pipeline panel to open its small Redo-line window for the selected
--   chunk; starts the panel first when it is not running.
--
--   The request is a non-persistent ext state the panel polls every frame:
--   dub_pipeline/redo_req = a fresh timestamp. dub_pipeline/alive is the
--   panel's heartbeat (reaper.time_precise(), refreshed once a second); it is
--   how this script knows not to launch a second copy — re-running the panel's
--   own action would restart it (set_action_options(2)).

-- Only a dub chunk (an item on a "Dub Chunks…" or "Un sync" track — the
-- panel's V5.is_chunk_item rule) asks for the popup.
local item = reaper.GetSelectedMediaItem(0, 0)
local tn = ''
if item then
  local _, n = reaper.GetSetMediaTrackInfo_String(
    reaper.GetMediaItem_Track(item), 'P_NAME', '', false)
  tn = n or ''
end
if tn:sub(1, 10) ~= 'Dub Chunks' and tn ~= 'Un sync' then
  reaper.MB('Select a dub chunk first (an item on a "Dub Chunks" track), ' ..
            'then press the shortcut again.', 'Dub: Redo this line', 0)
  return
end

reaper.SetExtState('dub_pipeline', 'redo_req',
                   string.format('%.3f', reaper.time_precise()), false)

local alive = tonumber(reaper.GetExtState('dub_pipeline', 'alive')) or 0
if reaper.time_precise() - alive < 3 then return end

-- Not running: start it. The request above is already waiting for it.
local sep   = package.config:sub(1, 1)
local here  = (debug.getinfo(1, 'S').source:match('@(.+)') or ''):match('(.+)[/\\]') or '.'
local panel = here .. sep .. 'Dub_Pipeline_Panel.lua'
local f = io.open(panel, 'r')
if not f then
  reaper.MB('Dub_Pipeline_Panel.lua was not found next to this script:\n' .. here,
            'Dub: Redo this line', 0)
  return
end
f:close()
if reaper.AddRemoveReaScript then
  local ok, cmd = pcall(reaper.AddRemoveReaScript, true, 0, panel, true)
  if ok and cmd and cmd ~= 0 then reaper.Main_OnCommand(cmd, 0) return end
end
dofile(panel)
