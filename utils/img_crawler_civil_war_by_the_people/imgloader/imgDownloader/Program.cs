using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

namespace imgDownloader
{
    class Program
    {
        static void Main(string[] args)
        {
            List<string> itemurl = new List<string>();
            List<string> picUrlBase = new List<string>();
            List<string> itemName = new List<string>();

            string baseurl = "https://crowd.loc.gov";
            string url = "https://crowd.loc.gov/topics/civil-war/";
            HttpWebRequest myRequest = (HttpWebRequest)WebRequest.Create(url);
            myRequest.Method = "GET";
            WebResponse myResponse = myRequest.GetResponse();
            StreamReader sr = new StreamReader(myResponse.GetResponseStream(), System.Text.Encoding.UTF8);
            string result = sr.ReadToEnd();
            sr.Close();
            myResponse.Close();

            string[] lines = result.Split(new string[] { "\n" }, StringSplitOptions.None);
            int count = 0;

            List<string> gourps = new List<string>();
            foreach (var line in lines)
            {
                //Console.WriteLine(line.Length.ToString());
                if (line.Contains("href=\"/campaigns/"))
                {
                    //Console.WriteLine(line);

                    gourps.Add(line);
                }
            }
            List<string> gplinks = new List<string>();
            bool skiper = false;
            foreach (var item in gourps)
            {
                if (skiper)
                {
                    skiper = false;
                    continue;
                }
                skiper = true;
                string[] str = item.Split(new string[] { "\"" }, StringSplitOptions.RemoveEmptyEntries);
                bool getit = false;

                foreach (var part in str)
                {
                    if (getit)
                    {
                        gplinks.Add(baseurl + part);
                        //Console.WriteLine(gplinks[gplinks.Count - 1]);

                        break;
                    }
                    if (part.Contains("<a href="))
                    {
                        //Console.WriteLine(part);
                        getit = true;
                    }

                }
            }

            List<string> seriesurl = new List<string>();
            List<string> seriesname = new List<string>();
            foreach (var link in gplinks)
            {
                int pid = 1;
                while (true)
                {
                    try
                    {
                        myRequest = (HttpWebRequest)WebRequest.Create(link + "page=" + pid.ToString());
                        myRequest.Method = "GET";
                        myResponse = myRequest.GetResponse();
                        sr = new StreamReader(myResponse.GetResponseStream(), System.Text.Encoding.UTF8);
                        result = sr.ReadToEnd();
                        sr.Close();
                        myResponse.Close();

                        string[] slines = result.Split(new string[] { "\n" }, StringSplitOptions.None);
                        List<string> sgourps = new List<string>();
                        List<string> sgname = new List<string>();

                        bool getnext = false;
                        foreach (var line in slines)
                        {
                            if (getnext)
                            {

                                //Console.WriteLine(line);
                                sgname.Add(line);
                                getnext = false;
                            }
                            //Console.WriteLine(line.Length.ToString());
                            if (line.Contains("a href=\"/campaigns/"))
                            {

                                //Console.WriteLine(line);
                                getnext = true;
                                sgourps.Add(line);
                            }
                        }

                        int len = sgourps.Count;

                        for (int i = 0; i < len; i += 2)
                        {
                            string[] sg = sgourps[i].Split(new string[] { "\"" }, StringSplitOptions.None);
                            string[] sgn = sgname[i].Split(new string[] { "\"" }, StringSplitOptions.None);
                            bool getsurl = false;
                            bool getsname = false;

                            foreach (var item in sg)
                            {
                                if (getsurl)
                                {
                                    itemurl.Add(item);
                                    getsurl = false;
                                }
                                if (item.Contains("a href="))
                                {
                                    getsurl = true;
                                }
                            }

                            foreach (var item in sgn)
                            {
                                if (getsname)
                                {
                                    itemName.Add(item);
                                    getsname = false;
                                }
                                if (item.Contains("alt="))
                                {
                                    getsname = true;
                                }
                            }
                            //count++;
                        }

                        Console.WriteLine(itemurl[itemurl.Count - 1]);
                    }
                    catch (Exception ex)
                    {
                        if (ex.Message.Contains("404"))
                        {
                            Console.WriteLine(ex.Message);
                            break;
                        }
                        
                    }
                    pid++;
                }
            }

            // construct local path
            List<string> itemPath = new List<string>();

            // url construction
            string picServ = "https://crowd-media.loc.gov/";
            string basepath = "campaigns/";
            int nIt = itemurl.Count;
            for (int i = 0; i < nIt; i++)
            {
                String first_slash_rm = itemurl[i].Substring(1);
                int startid = first_slash_rm.IndexOf('/') + 1;
                int lastid = first_slash_rm.LastIndexOf('?');
                string trimed = first_slash_rm.Substring(startid, lastid - startid);
                itemPath.Add(basepath + trimed);
                picUrlBase.Add(picServ + trimed);
            }

            for (int i = 0; i < nIt; i++)
            {
                itemName[i] = itemName[i].Replace(',', '|');
            }

            foreach (var path in itemPath)
            {
                Directory.CreateDirectory(path);
            }

            // construct csv to store title info
            StringBuilder sb = new StringBuilder();
            sb.Append("title,url,local path,years\n");
            for (int i = 0; i < nIt; i++)
            {
                string[] years = Regex.Matches(itemName[i], @"\d{4}").Cast<Match>().Select(m => m.Value).ToArray();
                string yr = String.Join(",", years);
                sb.Append(String.Join(",", new string[] { itemName[i], picUrlBase[i], itemPath[i], yr }) + "\n");
            }
            using (System.IO.StreamWriter file =
            new System.IO.StreamWriter("civil-war-images-info.csv", false))
            {
                file.WriteLine(sb.ToString());
            }


            //Console.WriteLine(sb.ToString());

            for (int i = 0; i < nIt; i++)
            {
                int iter = 1;
                while (true)
                {
                    try
                    {
                        using (WebClient webClient = new WebClient())
                        {
                            while (true)
                            {
                                string picurl = picUrlBase[i] + iter.ToString() + ".jpg";
                                string localpath = itemPath[i] + iter.ToString() + ".jpg";
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
                        Console.WriteLine(picUrlBase[i] + iter.ToString() + ".jpg");
                        iter++;
                        count++;
                    }
                    catch (Exception ex)
                    {
                        if (!ex.Message.Contains("404"))
                        {
                            break;
                            
                        }
                        Console.WriteLine(ex.Message);
                    }
                }
            }


            //Console.WriteLine(String.Join("\n", itemurl.ToArray()));
            //Console.WriteLine(String.Join("\n", itemName.ToArray()));
            //Console.WriteLine(String.Join("\n", itemPath.ToArray()));
            //Console.WriteLine(String.Join("\n", picUrlBase.ToArray()));
            //Console.WriteLine(gplinks.Count.ToString());
            Console.WriteLine(count.ToString());
            //Console.Write(result);

            Console.ReadKey();
        }
    }
}
