using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace itemTypeLabeler
{
    public partial class Form1 : Form
    {
        private List<string> filelist,shuffled,microfilms,scans,labeled,uncertains;
        private List<int> undos;
        private int seed;
        private Random rand;
        private int total;

        private int counter;
        private string cur;

        private void btn_sv_Click(object sender, EventArgs e)
        {
            using (TextWriter tw = new StreamWriter("microfilms.txt"))
            {
                tw.WriteLine(String.Join("\n", microfilms.ToArray()));
            }
            using (TextWriter tw = new StreamWriter("scans.txt"))
            {
                tw.WriteLine(String.Join("\n", scans.ToArray()));
            }
            using (TextWriter tw = new StreamWriter("uncertains.txt"))
            {
                tw.WriteLine(String.Join("\n", uncertains.ToArray()));
            }
            using (TextWriter tw = new StreamWriter("shuffledlist.txt"))
            {
                tw.WriteLine(String.Join("\n", shuffled.ToArray()));
            }
            using (TextWriter tw = new StreamWriter("labeledlist.txt"))
            {
                tw.WriteLine(String.Join("\n", labeled.ToArray()));
            }
        }

        private void btn_undo_Click(object sender, EventArgs e)
        {
            int last = undos.Last();
            switch (last) {
                case 1:
                    microfilms.RemoveAt(microfilms.Count - 1);
                    break;
                case 2:
                    scans.RemoveAt(scans.Count - 1);
                    break;
                case 3:
                    uncertains.RemoveAt(uncertains.Count - 1);
                    break;
            }
            cur = labeled.Last();
            labeled.RemoveAt(labeled.Count - 1);
            undos.RemoveAt(undos.Count - 1);

            counter--;

            this.nbc.Text = (counter).ToString();
            this.nblf.Text = (total - counter).ToString();
            this.nbmc.Text = microfilms.Count.ToString();
            this.nbsc.Text = scans.Count.ToString();
            this.nbsc.Text = scans.Count.ToString();

            try
            {
                this.pb_show.Image = Image.FromFile(cur);
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message);
            }
        }

        private void btn_mc_Click(object sender, EventArgs e)
        {
            microfilms.Add(cur);
            labeled.Add(cur);

            try
            {
                cur = shuffled[counter];
                this.pb_show.Image = Image.FromFile(cur);
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message);
            }

            counter++;
            this.nbc.Text = (counter).ToString();
            this.nblf.Text = (total - counter).ToString();
            this.nbmc.Text = microfilms.Count.ToString();

            undos.Add(1);
        }

        private void btn_sc_Click(object sender, EventArgs e)
        {
            scans.Add(cur);
            labeled.Add(cur);
            try
            {
                cur = shuffled[counter];
                this.pb_show.Image = Image.FromFile(cur);
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message);
            }

            counter++;
            this.nbc.Text = (counter).ToString();
            this.nblf.Text = (total - counter).ToString();
            this.nbsc.Text = scans.Count.ToString();
            
            undos.Add(2);
        }

        private void btn_un_Click(object sender, EventArgs e)
        {
            uncertains.Add(cur);
            labeled.Add(cur);

            try
            {
                cur = shuffled[counter];
                this.pb_show.Image = Image.FromFile(cur);
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message);
            }

            counter++;
            this.nbc.Text = (counter).ToString();
            this.nblf.Text = (total - counter).ToString();
            this.nbun.Text = uncertains.Count.ToString();

            undos.Add(3);
        }

        public Form1()
        {
            InitializeComponent();
        }

        private void Form1_Load(object sender, EventArgs e)
        {
            undos = new List<int>();

            filelist = new List<string>();
            shuffled = new List<string>();
            microfilms = new List<string>();
            scans = new List<string>();
            labeled = new List<string>();

            uncertains = new List<string>();

            seed = 1;
            rand = new Random(seed);

            string line;
            try
            {
                // Read the file and display it line by line.  
                System.IO.StreamReader file =
                    new System.IO.StreamReader("filelist.txt");
                while ((line = file.ReadLine()) != null)
                {
                    filelist.Add(line);
                    
                }

                file.Close();
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message);
            }
            shuffled = ShuffleList(filelist, rand);

            counter = 1;
            total = shuffled.Count;
            this.nbt.Text = total.ToString();
            this.nblf.Text = (total - counter).ToString();
            this.nbc.Text = "1";

            this.nbmc.Text = "0";
            this.nbsc.Text = "0";
            this.nbun.Text = "0";

            try
            {
                cur = shuffled[0];
                this.pb_show.Image = Image.FromFile(cur);
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message);
            }
            
        }

        private List<E> ShuffleList<E>(List<E> inputList, Random rand)
        {
            List<E> randomList = new List<E>();
            int randomIndex = 0;
            while (inputList.Count > 0)
            {
                randomIndex = rand.Next(0, inputList.Count); //Choose a random object in the list
                randomList.Add(inputList[randomIndex]); //add it to the new, random list
                inputList.RemoveAt(randomIndex); //remove to avoid duplicates
            }

            return randomList; //return the new random list
        }
    }
}
