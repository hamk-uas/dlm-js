function save_json(obj, filename)
  fid = fopen (filename, "w");
  str = jsonencode(obj, "PrettyPrint", true);
  fputs(fid, str);
  fclose (fid);
end
