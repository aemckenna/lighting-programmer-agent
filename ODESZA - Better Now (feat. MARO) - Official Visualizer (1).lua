--********************************************************************
--**           Lighting Cue Importer                                **
--********************************************************************
local CMD = gma.cmd;
local ECHO = gma.echo;
local FEEDBACK = gma.feedback;
local DIALOG = gma.gui.confirm;
local GET_OBJ = gma.show.getobj
local GET_HANDLE = gma.show.getobj.handle;
local GET_INDEX = gma.show.getobj.index;
local GET_PROPERTY = gma.show.property.get;
local GETVAR = gma.show.getvar;
local ZZZ = gma.sleep;
local LUA_NAME = 'LightingCues';

--********************************************************************
--**             Main Function                                      **
--********************************************************************
local function lightingcueimporter()

-- Store Cues:
CMD('Store Sequence 23 Cue 1 "Verse 0-19"');
CMD('Assign Sequence 23 Cue 1 /Fade=3.0');
CMD('Store Sequence 23 Cue 2 "Verse 19-35"');
CMD('Assign Sequence 23 Cue 2 /Fade=3.0');
CMD('Store Sequence 23 Cue 3 "Pre-Chorus 35-51"');
CMD('Assign Sequence 23 Cue 3 /Fade=3.0');
CMD('Store Sequence 23 Cue 4 "Chorus 51-68"');
CMD('Assign Sequence 23 Cue 4 /Fade=3.0');
CMD('Store Sequence 23 Cue 5 "Chorus 68-84"');
CMD('Assign Sequence 23 Cue 5 /Fade=3.0');
CMD('Store Sequence 23 Cue 6 "Chorus 84-104"');
CMD('Assign Sequence 23 Cue 6 /Fade=3.0');
CMD('Store Sequence 23 Cue 7 "Verse 104-125"');
CMD('Assign Sequence 23 Cue 7 /Fade=3.0');
CMD('Store Sequence 23 Cue 8 "Verse 125-147"');
CMD('Assign Sequence 23 Cue 8 /Fade=3.0');
CMD('Store Sequence 23 Cue 9 "Pre-Chorus 147-166"');
CMD('Assign Sequence 23 Cue 9 /Fade=3.0');
CMD('Store Sequence 23 Cue 10 "Chorus 166-185"');
CMD('Assign Sequence 23 Cue 10 /Fade=3.0');
CMD('SelectDrive 1');
gma.sleep(0.5);
end
return lightingcueimporter;