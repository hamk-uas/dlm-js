function obj = load_json(filename)
  fid = fopen (filename, "r");
  raw = fread(fid, inf);
  str = char(raw');
  obj = jsondecode(str);
  fclose (fid);
end