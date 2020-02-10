using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace missingImgFinder
{
    class Program
    {
        static void Main(string[] args)
        {
            int counter = 0;
            string line;
            List<string> skipedLines = new List<string>();
            List<string> lines = new List<string>();
            List<int> ids = new List<int>();

            List<string> skippedPaths = new List<string>();
            string basePath = "campaigns/";

            List<string> largest = new List<string>();

            // Read the file and display it line by line.  
            System.IO.StreamReader file =
                new System.IO.StreamReader("trimed_logs.txt");
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
                if(thisid != 1)
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
                    string baseurl = lastline.Substring(0, lastline.LastIndexOf('/') + 1);
                    string largeurl = baseurl + (lastid + 1).ToString() + ".jpg";
                    largest.Add(largeurl);
                    counter++;
                }

                lastline = thisline;
                lastid = thisid;
            }

            Console.WriteLine(String.Join("\n", skipedLines.ToArray()));
            Console.WriteLine(String.Join("\n", skippedPaths.ToArray()));
            //Console.WriteLine(String.Join("\n", largest.ToArray()));
            //System.Console.WriteLine("There were {0} lines.", counter);

            // Suspend the screen.  
            System.Console.ReadLine();
        }
    }
}
