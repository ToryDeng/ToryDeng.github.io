// get the ninja-keys element
const ninja = document.querySelector('ninja-keys');

// add the home and posts menu items
ninja.data = [{
    id: "nav-about",
    title: "about",
    section: "Navigation",
    handler: () => {
      window.location.href = "/";
    },
  },{id: "nav-blog",
          title: "blog",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/blog/";
          },
        },{id: "nav-publications",
          title: "publications",
          description: "*: equal contribution.",
          section: "Navigation",
          handler: () => {
            window.location.href = "/publications/";
          },
        },{id: "nav-repositories",
          title: "repositories",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/repositories/";
          },
        },{id: "nav-teaching",
          title: "teaching",
          description: "I mainly served as a teaching assistant for statistics and applied mathematics courses.",
          section: "Navigation",
          handler: () => {
            window.location.href = "/teaching/";
          },
        },{id: "nav-cv-english",
          title: "CV (English)",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/assets/pdf/English_Resume.pdf";
          },
        },{id: "nav-cv-chinese",
          title: "CV (Chinese)",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/assets/pdf/Chinese_Resume.pdf";
          },
        },{id: "post-the-perron-frobenius-theorem",
      
        title: "The Perron-Frobenius Theorem",
      
      description: "a detailed proof of the PF theorem and an application",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/2023/perron-frobenius/";
        
      },
    },{id: "post-deploying-a-server-for-bioinformatics-research",
      
        title: "Deploying a Server for Bioinformatics Research",
      
      description: "how to deploy a server for bioinformatics research",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/2023/deploying-server/";
        
      },
    },{id: "news-our-research-on-cofunctional-gene-groups-across-both-cell-types-and-tissue-domains-has-been-accepted-in-gpb-sparkles-smile",
          title: 'Our research on cofunctional gene groups across both cell types and tissue domains...',
          description: "",
          section: "News",},{id: "projects-project-7",
          title: 'project 7',
          description: "with background image",
          section: "Projects",handler: () => {
              window.location.href = "/projects/7_project/";
            },},{id: "projects-project-8",
          title: 'project 8',
          description: "an other project with a background image and giscus comments",
          section: "Projects",handler: () => {
              window.location.href = "/projects/8_project/";
            },},{id: "projects-project-9",
          title: 'project 9',
          description: "another project with an image 🎉",
          section: "Projects",handler: () => {
              window.location.href = "/projects/9_project/";
            },},{
        id: 'social-email',
        title: 'email',
        section: 'Socials',
        handler: () => {
          window.open("mailto:%74%61%6F%64%65%6E%67@%6C%69%6E%6B.%63%75%68%6B.%65%64%75.%63%6E", "_blank");
        },
      },{
        id: 'social-github',
        title: 'GitHub',
        section: 'Socials',
        handler: () => {
          window.open("https://github.com/ToryDeng", "_blank");
        },
      },{
        id: 'social-orcid',
        title: 'ORCID',
        section: 'Socials',
        handler: () => {
          window.open("https://orcid.org/0000-0001-7401-311X", "_blank");
        },
      },{
        id: 'social-rss',
        title: 'RSS Feed',
        section: 'Socials',
        handler: () => {
          window.open("/feed.xml", "_blank");
        },
      },{
        id: 'social-scholar',
        title: 'Google Scholar',
        section: 'Socials',
        handler: () => {
          window.open("https://scholar.google.com/citations?user=0QI92MsAAAAJ", "_blank");
        },
      },{
      id: 'light-theme',
      title: 'Change theme to light',
      description: 'Change the theme of the site to Light',
      section: 'Theme',
      handler: () => {
        setThemeSetting("light");
      },
    },
    {
      id: 'dark-theme',
      title: 'Change theme to dark',
      description: 'Change the theme of the site to Dark',
      section: 'Theme',
      handler: () => {
        setThemeSetting("dark");
      },
    },
    {
      id: 'system-theme',
      title: 'Use system default theme',
      description: 'Change the theme of the site to System Default',
      section: 'Theme',
      handler: () => {
        setThemeSetting("system");
      },
    },];
