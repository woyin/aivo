use super::super::*;
use serde_json::json;

#[test]
fn remote_mutation_flags_outward_writes() {
    // HTTP clients: a mutating method or a request body.
    assert!(bash_mutates_remote("curl -X POST https://api/x -d '{}'"));
    assert!(bash_mutates_remote("curl -XDELETE https://api/x"));
    assert!(bash_mutates_remote("curl --request PUT https://api/x"));
    assert!(bash_mutates_remote(
        "curl -F file=@a.zip https://api/upload"
    ));
    assert!(bash_mutates_remote("curl -T a.txt https://api/put"));
    assert!(bash_mutates_remote("curl --json '{}' https://api/x"));
    assert!(bash_mutates_remote("wget --method=DELETE https://api/x"));
    assert!(bash_mutates_remote("http POST example.com key=val"));
    // Cloud / infra / deploy CLIs.
    assert!(bash_mutates_remote("gh repo delete owner/name --yes"));
    assert!(bash_mutates_remote("gh release create v1"));
    assert!(bash_mutates_remote("gh api repos/o/r -X DELETE"));
    assert!(bash_mutates_remote("gh api repos/o/r/issues -f title=x"));
    assert!(bash_mutates_remote("aws s3 rm s3://bucket/key"));
    assert!(bash_mutates_remote(
        "aws ec2 terminate-instances --instance-ids i-1"
    ));
    assert!(bash_mutates_remote(
        "aws ec2 run-instances --image-id ami-1"
    ));
    assert!(bash_mutates_remote("gcloud compute instances create vm-1"));
    assert!(bash_mutates_remote("gcloud app deploy"));
    assert!(bash_mutates_remote("az group delete --name rg"));
    assert!(bash_mutates_remote("az webapp up"));
    assert!(bash_mutates_remote("kubectl delete pod x"));
    assert!(bash_mutates_remote("kubectl apply -f d.yaml"));
    assert!(bash_mutates_remote("kubectl rollout restart deploy/x"));
    assert!(bash_mutates_remote("helm upgrade rel chart"));
    assert!(bash_mutates_remote("helm uninstall rel"));
    assert!(bash_mutates_remote("terraform apply -auto-approve"));
    assert!(bash_mutates_remote("terraform destroy"));
    assert!(bash_mutates_remote("docker push repo/img:tag"));
    assert!(bash_mutates_remote("npm publish"));
    assert!(bash_mutates_remote("cargo publish"));
    assert!(bash_mutates_remote("vercel deploy --prod"));
    assert!(bash_mutates_remote("flyctl deploy"));
    assert!(bash_mutates_remote("railway up"));
    // See-through wrappers.
    assert!(bash_mutates_remote("sudo kubectl delete ns team"));
    assert!(bash_mutates_remote("sh -c 'curl -X POST https://api/x'"));
    // Any segment in a pipeline / chain.
    assert!(bash_mutates_remote(
        "cat body.json | curl -X POST -d @- https://api/x"
    ));
}

#[test]
fn remote_mutation_leaves_reads_alone() {
    // Plain GETs / downloads.
    assert!(!bash_mutates_remote("curl https://example.com"));
    assert!(!bash_mutates_remote("curl -fsSL https://example.com/x")); // -f is --fail, not --form
    assert!(!bash_mutates_remote("curl -X GET https://api/x"));
    assert!(!bash_mutates_remote("curl -G -d q=1 https://api/search")); // -G ⇒ GET query
    assert!(!bash_mutates_remote("wget https://example.com/file.tgz"));
    // Read-only cloud queries.
    assert!(!bash_mutates_remote("gh pr list"));
    assert!(!bash_mutates_remote("gh run list"));
    assert!(!bash_mutates_remote("gh repo view owner/name"));
    assert!(!bash_mutates_remote("gh api repos/o/r"));
    assert!(!bash_mutates_remote("aws s3 ls s3://bucket"));
    assert!(!bash_mutates_remote("aws ec2 describe-instances"));
    assert!(!bash_mutates_remote("gcloud compute instances list"));
    assert!(!bash_mutates_remote("az account show"));
    assert!(!bash_mutates_remote("kubectl get pods"));
    assert!(!bash_mutates_remote("kubectl rollout status deploy/x"));
    assert!(!bash_mutates_remote("helm list"));
    assert!(!bash_mutates_remote("helm create mychart")); // local scaffold
    assert!(!bash_mutates_remote("terraform plan"));
    assert!(!bash_mutates_remote("docker ps"));
    assert!(!bash_mutates_remote("docker build -t x ."));
    assert!(!bash_mutates_remote("docker rm container")); // local
    assert!(!bash_mutates_remote("npm install")); // local download
    assert!(!bash_mutates_remote("cargo build"));
    assert!(!bash_mutates_remote("git push")); // git handled by the destructive walk
    // Local file work that shares a verb word.
    assert!(!bash_mutates_remote("rm -rf ./build"));
    assert!(!bash_mutates_remote("ls -la"));
    // Public wrapper only fires for run_bash.
    assert!(is_remote_side_effect(
        "run_bash",
        &json!({ "command": "gh repo delete o/r" })
    ));
    assert!(!is_remote_side_effect(
        "run_bash",
        &json!({ "command": "gh pr list" })
    ));
    assert!(!is_remote_side_effect(
        "write_file",
        &json!({ "path": "a.txt", "content": "" })
    ));
}

#[test]
fn remote_prefixes_name_the_family_up_to_the_verb() {
    assert_eq!(
        remote_mutation_prefixes("az repos pr update --id 7 --status completed"),
        vec!["az repos pr update"]
    );
    assert_eq!(
        remote_mutation_prefixes("gh pr merge 42 --squash"),
        vec!["gh pr merge"]
    );
    assert_eq!(
        remote_mutation_prefixes("aws ec2 terminate-instances --instance-ids i-1"),
        vec!["aws ec2 terminate-instances"]
    );
    assert_eq!(
        remote_mutation_prefixes("docker push repo/img:tag"),
        vec!["docker push"]
    );
    // The prefix survives a compound command: only the mutating segment names it.
    assert_eq!(
        remote_mutation_prefixes(
            "cd /repo && BASE=$(git merge-base HEAD origin/dev) && \
             az boards work-item create --title \"fix: x\" --type Task"
        ),
        vec!["az boards work-item create"]
    );
    // See-through wrappers, including an interpreter's inline program.
    assert_eq!(
        remote_mutation_prefixes("sudo kubectl delete ns team"),
        vec!["kubectl delete"]
    );
    assert_eq!(
        remote_mutation_prefixes("sh -c 'az repos pr update --id 7'"),
        vec!["az repos pr update"]
    );
    // Duplicated families collapse to one grant.
    assert_eq!(
        remote_mutation_prefixes("gh pr merge 1 && gh pr merge 2"),
        vec!["gh pr merge"]
    );
}

#[test]
fn remote_prefixes_refuse_opaque_mutations() {
    // Flag-driven mutations have no verb path — exact grant only.
    assert!(remote_mutation_prefixes("curl -X POST https://api/x -d '{}'").is_empty());
    assert!(remote_mutation_prefixes("gh api repos/o/r/issues -f title=x").is_empty());
    // One opaque segment poisons the whole command.
    assert!(
        remote_mutation_prefixes("az repos pr update --id 7 && curl -X POST https://api/x")
            .is_empty()
    );
    assert!(remote_mutation_prefixes("gh pr list").is_empty());
    assert!(remote_mutation_prefixes("az account show").is_empty());
}
