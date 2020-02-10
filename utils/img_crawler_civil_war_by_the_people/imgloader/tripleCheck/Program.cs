using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace tripleCheck
{
    class Program
    {
        static void Main(string[] args)
        {
            int counter = 0;
            string line;

            List<string> skipedLines = new List<string>();
            List<string> skippedPaths = new List<string>();
            string basePath = "campaigns";

            List<int> ids = new List<int>();
            List<string> lines = new List<string>();

            List<string> largest = new List<string>();
            List<int> nbfile = new List<int>();

            // Read the file and display it line by line.  
            System.IO.StreamReader file =
                new System.IO.StreamReader("filelist.txt");
            while ((line = file.ReadLine()) != null)
            {
                //System.Console.WriteLine(line);
                int startid = line.LastIndexOf('/') + 1;
                int endid = line.LastIndexOf('.');
                string substr = line.Substring(startid, endid - startid);
                int id = int.Parse(substr);

                ids.Add(id);
                lines.Add(line);
                //Console.WriteLine(substr);
                counter++;
            }

            file.Close();
            System.Console.WriteLine("There were {0} lines.", counter);

            counter = 0;
            int iter = ids.Count;
            string lastline = lines[0];
            int lastid = ids[0];
            for (int i = 1; i < iter; i++)
            {
                int thisid = ids[i];
                string thisline = lines[i];
                if (thisid != 1)
                {
                    if (thisid - lastid != 1)
                    {
                        string baseurl = lastline.Substring(0, lastline.LastIndexOf('/') + 1);
                        string missingurl = baseurl + (lastid + 1).ToString() + ".jpg";
                        skipedLines.Add(missingurl);

                        string skippedpath = basePath + missingurl.Substring(missingurl.IndexOf("gov/") + 4);
                        skippedPaths.Add(skippedpath);
                        //counter++;
                    }
                }

                if (thisid == 1)
                {
                    string folderpath = lastline.Substring(0, lastline.LastIndexOf('/') + 1);
                    folderpath = folderpath.Replace(".", basePath);
                    largest.Add(folderpath);
                    nbfile.Add(lastid);
                    counter++;
                }

                lastline = thisline;
                lastid = thisid;
            }

            List<string> problems = new List<string>();
            int it = largest.Count;
            for (int i = 0; i < it; i++)
            {
                string path = largest[i];
                int nbf = nbfile[i];
                int fCount = Directory.GetFiles(path, "*.jpg", SearchOption.TopDirectoryOnly).Length;
                if (nbf != fCount)
                {
                    problems.Add(path);
                }
            }

            //Console.WriteLine(String.Join("\n", skipedLines.ToArray()));
            //Console.WriteLine(String.Join("\n", skippedPaths.ToArray()));
            Console.WriteLine(String.Join("\n", problems.ToArray()));
            System.Console.WriteLine("There were {0} lines.", counter);

            // Suspend the screen.  
            System.Console.ReadLine();
        }
    }
}
