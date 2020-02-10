using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net;
using System.Text;
using System.Threading.Tasks;

namespace doubleChecker
{
    class Program
    {
        static void Main(string[] args)
        {
            int counter = 0;
            
            string basePath = "campaigns/";

            string line;
            System.IO.StreamReader file =
                new System.IO.StreamReader("doubleCheck.txt");
            while ((line = file.ReadLine()) != null)
            {
                string startpath = basePath + line.Substring(line.IndexOf("gov/") + 4);
                
                int startid = line.LastIndexOf('/') + 1;
                int endid = line.LastIndexOf('.');
                string substr = line.Substring(startid, endid - startid);
                int id = int.Parse(substr);

                string folderpath = startpath.Substring(0, startpath.LastIndexOf('/') + 1);
                string baseurl = line.Substring(0, line.LastIndexOf('/') + 1);
                Directory.CreateDirectory(folderpath);

                counter++;
                Console.WriteLine(counter.ToString());
                Console.WriteLine(baseurl);

                int iter = id + 1;
                while (true)
                {
                    try
                    {
                        using (WebClient webClient = new WebClient())
                        {
                            while (true)
                            {
                                string picurl = baseurl + iter.ToString() + ".jpg";
                                string localpath = folderpath + iter.ToString() + ".jpg";
                                //throw new Exception("manual fail");
                                webClient.DownloadFile(picurl, localpath);
                                // Obtain the WebHeaderCollection instance containing the header name/value pair from the response.
                                WebHeaderCollection webHeaderCollection = webClient.ResponseHeaders;
                                long remoteFileSize = long.Parse(webHeaderCollection.GetValues("Content-Length")[0]);
                                FileInfo localFile = new FileInfo(localpath);
                                long localFileSize = localFile.Length;
                                if (remoteFileSize == localFileSize)
                                {
                                    break;
                                }
                            }
                        }
                        Console.WriteLine(baseurl + iter.ToString() + ".jpg");
                        iter++;
                    }
                    catch (Exception ex)
                    {
                        if (ex.Message.Contains("404"))
                        {
                            break;
                        }
                        Console.WriteLine(ex.Message);
                    }
                }
            }

            file.Close();
            Console.ReadKey();
        }
    }
}
