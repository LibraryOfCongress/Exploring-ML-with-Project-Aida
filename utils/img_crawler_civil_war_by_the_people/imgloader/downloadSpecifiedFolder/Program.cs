using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net;
using System.Text;
using System.Threading.Tasks;

namespace downloadSpecifiedFolder
{
    class Program
    {
        static void Main(string[] args)
        {
            if(args.Length != 1)
            {
                Console.WriteLine("Need base path");
                return;
            }
            string local = args[0];
            string itemPath = local.Substring(local.IndexOf('/') + 1) + "/";
            string picUrlBase = "https://crowd-media.loc.gov/" + itemPath.Substring(itemPath.IndexOf('/') + 1);

            Directory.CreateDirectory(itemPath);

            int iter = 1;
            while (true)
            {
                try
                {
                    using (WebClient webClient = new WebClient())
                    {
                        while (true)
                        {
                            string picurl = picUrlBase + iter.ToString() + ".jpg";
                            string localpath = itemPath + iter.ToString() + ".jpg";
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
                    Console.WriteLine(picUrlBase + iter.ToString() + ".jpg");
                    iter++;
                }
                catch (Exception ex)
                {
                    if (!ex.Message.Contains("404"))
                    {
                        Console.WriteLine(ex.Message);
                    }
                    break;
                }
            }
            Console.ReadKey();
        }
    }
}
