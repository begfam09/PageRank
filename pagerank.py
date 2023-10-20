import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    print(corpus)
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    trans_model = {}

    # If this page has no linked pages, all the pages are equally likely to be next
    if len(corpus[page]) == 0:
        for local_page in corpus:
            trans_model[local_page] = 1 / len(corpus)

    # This page does have links. Need to determine probabilities of next page given outbound links
    # present and the possibility surfer could go to random page next.
    else:
        prob_per_link = damping_factor / len(corpus[page])
        prob_per_page = (1 - damping_factor) / len(corpus)

        for local_page in corpus:
            if local_page in corpus[page]:
                trans_model[local_page] = prob_per_page + prob_per_link
            else:
                trans_model[local_page] = prob_per_page

    return trans_model


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """

    # Set initial PageRanks to 0
    pagerank_dict = {page: 0 for page in corpus}

    # We need a random starting page
    initial_page = random.choice(list(corpus.keys()))

    # Given a starting page, see what page the surfer goes to next, using the probabilities returned
    # by the transition model. Keep track of what page is visited. Do this 'n' times.
    for _ in range(n):
        model = transition_model(corpus, initial_page, damping_factor)
        next_page = random.choices(list(model.keys()), weights=model.values(), k=1)[0]
        pagerank_dict[next_page] += 1
        initial_page = next_page

    # To get percentages that add up to 1, divide the number of times each page was visited by the
    # total number of samples.
    for page in pagerank_dict:
        pagerank_dict[page] /= n

    return pagerank_dict


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank (PR) values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # Set initial PRs to equal probability
    pagerank_dict = {page: 1/len(corpus) for page in corpus}

    # If a page has no links, we will treat it as if it has a link to every page
    for page, links in corpus.items():
        if not links:
            corpus[page] = set(corpus.keys())

    # First half of PR equation - small chance surfer will go to random page instead of follow links
    chose_page_at_random = (1 - damping_factor) / len(corpus)

    # Calculate updated PR for each page in the corpus, using the PRs of all pages that link to it. Do
    # this until the difference between the all previous & new PRs is less than .001.
    while True:
        new_ranks = {}
        for page in corpus:
            rank_sum = sum(pagerank_dict[link] / len(corpus[link]) for link in corpus if page in corpus[link])
            new_ranks[page] = chose_page_at_random + (damping_factor * rank_sum)

        if all(abs(new_ranks[page] - pagerank_dict[page]) < 0.001 for page in pagerank_dict):
            break

        pagerank_dict = new_ranks

    return pagerank_dict


if __name__ == "__main__":
    main()
