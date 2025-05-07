# LightricksCandidateProject
A home assignment of designing an advanced image editing system which is capable of applying custom filters and adjustments to images

The JSON file used for operation data should look like this:
{
 "input": "string (required, path to input image)",
 "output": "string (optional, path to save output image)",
 "display": "boolean (optional, whether to display the final image)",
 "operations": [
  {
   "type": "string (required)",
   "<parameter_key>": "<parameter_value>"
  }
 ]
}
