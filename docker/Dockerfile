FROM ubuntu:latest

WORKDIR /sw

# Build a base Docker
RUN apt-get update -y && \
    DEBIAN_FRONTEND="noninteractive" apt-get -y install tzdata && \
    apt-get install -y apt-transport-https ca-certificates curl software-properties-common && \
    curl -fsSL https://debian.neo4j.com/neotechnology.gpg.key | apt-key add - && \
    add-apt-repository "deb https://debian.neo4j.com stable 4.1" && \
    apt-get install -y neo4j && \
    sed -i 's/#dbms.default_database=neo4j/dbms.default_database=indra/g' /etc/neo4j/neo4j.conf && \
    sed -i 's/#dbms.default_listen_address=0.0.0.0/dbms.default_listen_address=0.0.0.0/g' /etc/neo4j/neo4j.conf

# Now add and import the data
ADD nodes.tsv.gz /sw/nodes.tsv.gz
ADD edges.tsv.gz /sw/edges.tsv.gz

RUN neo4j-admin import --database indra --delimiter='TAB' --skip-duplicate-nodes=true --skip-bad-relationships=true \
        --nodes /sw/nodes.tsv.gz --relationships /sw/edges.tsv.gz

ADD startup.sh /sw/startup.sh
ENTRYPOINT ["/sw/startup.sh"]
